import json
import logging
import os
import traceback
from typing import Dict, List, Optional, Tuple, Any

import ImageReward as reward
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from transformers import CLIPProcessor, CLIPModel

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
# from comfy.utils import ProgressBar

from .qwen_verifier import QwenVLMVerifier
from .utils import generate_neighbors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inferencescale')


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                    override_noise=None):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    if override_noise is not None:
        noise = override_noise.to(latent_image.device)
    elif disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    # callback = latent_preview.prepare_callback(model, steps)
    # disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    disable_pbar = True
    callback = None

    samples = comfy.sample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )

    out = latent.copy()
    out["samples"] = samples
    return (out, )


def rank_candidates(results_by_verifier, num_candidates):
    """
    results_by_verifier: {
       'clip': [score0, score1, ...], # Scores for each image from CLIP
       'image_reward': [...], # Scores from ImageReward
       'qwen_vlm_verifier': [...], # Scores from Qwen
       ...
    }
    Convert each verifier's raw scores into ranks (best=rank 1), then compute
    an average rank for each candidate, returning a list of dicts sorted by avg_rank.
    """
    ranks_by_verifier = {}
    for verifier_name, scores_list in results_by_verifier.items():
        # Sort indices by descending score (best score = rank 1)
        sorted_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        rank_array = [0] * len(scores_list)
        for rank, idx in enumerate(sorted_indices):
            rank_array[idx] = rank + 1  # best = rank 1
        ranks_by_verifier[verifier_name] = rank_array

    n_verifiers = len(results_by_verifier)
    final_results = []
    for i in range(num_candidates):
        sum_ranks = 0
        image_scores = {}
        for verifier_name, score_list in results_by_verifier.items():
            image_scores[verifier_name] = score_list[i]
            sum_ranks += ranks_by_verifier[verifier_name][i]
        avg_rank = (sum_ranks / n_verifiers) if n_verifiers > 0 else 9999
        final_results.append({"index": i, "avg_rank": avg_rank, "scores": image_scores})
    # Sort ascending by avg_rank (lower is better)
    final_results.sort(key=lambda x: x["avg_rank"])
    return final_results


def score_candidates(candidate_tensors: List[torch.Tensor], text_prompt: str, verifier_names: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Scores each candidate image using the provided verifier ensemble.
    
    Args:
        candidate_tensors: List of image tensors to score
        text_prompt: Text prompt to compare against
        verifier_names: Dictionary of verifier instances
        
    Returns:
        List of score dictionaries, one per candidate
    """
    num_candidates = len(candidate_tensors)
    raw_scores_list = [{} for _ in range(num_candidates)]

    try:
        if not candidate_tensors:
            raise ValueError("No candidate tensors provided for scoring")

        if not text_prompt:
            logger.warning("Empty text prompt provided for scoring")

        for verifier_name, verifier in verifier_names.items():
            if verifier is None:
                continue

            logger.info(f"Scoring {num_candidates} candidates with {verifier_name}")
            
            try:
                with torch.no_grad():
                    for i, tensor_img in enumerate(candidate_tensors):
                        try:
                            pil_img = ToPILImage()(tensor_img.squeeze(0).permute(2, 0, 1))
                            
                            if verifier_name == "clip":
                                score = verifier.score(text_prompt, pil_img)
                            elif verifier_name == "image_reward":
                                score = verifier.score(text_prompt, pil_img)
                            elif verifier_name == "qwen_vlm_verifier":
                                score = verifier.score(pil_img, text_prompt)
                            else:
                                logger.warning(f"Unknown verifier type: {verifier_name}")
                                continue
                                
                            raw_scores_list[i][verifier_name] = score
                            
                        except Exception as e:
                            logger.error(f"Error scoring candidate {i} with {verifier_name}: {str(e)}")
                            logger.debug(traceback.format_exc())
                            raw_scores_list[i][verifier_name] = float('-inf')
                            
            except Exception as e:
                logger.error(f"Error in verifier {verifier_name}: {str(e)}")
                logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"Fatal error in score_candidates: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

    return raw_scores_list


def rank_from_raw_scores(raw_scores_list):
    """
    Converts a list of raw score dictionaries into a ranked list.
    """
    # Build a verifier-to-scores dictionary from the raw_scores_list
    results_by_verifier = {}
    for candidate_scores in raw_scores_list:
        for verifier_name, score in candidate_scores.items():
            if verifier_name not in results_by_verifier:
                results_by_verifier[verifier_name] = []
            results_by_verifier[verifier_name].append(score)
    num_candidates = len(raw_scores_list)
    return rank_candidates(results_by_verifier, num_candidates)


class InferenceTimeScaler:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Denoising model."}),
                "vae": ("VAE", {"tooltip": "VAE model for decoding latents."}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of denoising steps to apply during each forward evaluation."}), 
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-Free Guidance Scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm to be used during each forward evaluation."}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise removal scheduler."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}), 
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Latent image to be denoised."}), 
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Amount of denoising."}),
                "search_algorithm": (["random", "zero-order"], {"default": "random", "tooltip": "Select the search algorithm: 'random' for standard random search or 'zero-order' for gradient-free local search."}),
                "text_prompt_to_compare": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt for verifier(s)."}), 
                "search_rounds": ("INT", {"default": 5, "min": 1, "max": 10000, "tooltip": "Number of search rounds (random seeds for random search, or iterations for zero-order search)."}),
                "view_top_k": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "Return grid view of the top-k images."}),
                "num_neighbors": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "Number of neighbors to sample per iteration in zero-order search (only used if search_algorithm is 'zero-order')."}),
                "lambda_threshold": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.001, "tooltip": "Perturbation step size for zero-order search (only used if search_algorithm is 'zero-order')."})
            },
            "optional": {
                "loaded_clip_score_verifier": ("CS_VERIFIER", {"tooltip": "HF CLIP model identifier."}),
                "loaded_image_reward_verifier": ("IR_VERIFIER", {"tooltip": "ImageReward model identifier."}),
                "loaded_qwen_verifier": ("QWN_VERIFIER", {"tooltip": "Qwen model identifier."})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Best Image", "Top-k Grid", "Top-k Score(s)")
    OUTPUT_TOOLTIPS = ("Best single image, grid of the top-k images, and their scores in JSON.",)
    FUNCTION = "execute"
    CATEGORY = "InferenceTimeScaling"
    DESCRIPTION = (
        "Performs inference-time optimization to find the best image matching your text prompt. "
        "Supports two search algorithms:\n"
        "1. Random Search: Generates multiple images with different random seeds\n"
        "2. Zero-Order Search: Performs gradient-free local optimization\n\n"
        "Uses an ensemble of AI verifiers (CLIP, ImageReward, Qwen-VL) to score and rank the generated images. "
        "Returns the best image, a grid of top-k results, and detailed scoring information."
    )

    def execute(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                text_prompt_to_compare, search_rounds, view_top_k, loaded_clip_score_verifier=None, loaded_image_reward_verifier=None,
                search_algorithm="random", num_neighbors=4, lambda_threshold=0.01, loaded_qwen_verifier=None):
        """
        Main execution method for inference time scaling.
        """
        try:
            # Only handling batch size of 1 for now
            if latent_image["samples"].size(0) != 1:
                raise ValueError(f"Expected latent image batch size of 1, got {latent_image['samples'].size(0)}")

            logger.info(f"Starting InferenceTimeScaler with algorithm: {search_algorithm}")
            logger.info(f"Search rounds: {search_rounds}, Top-k: {view_top_k}")

            if not text_prompt_to_compare:
                logger.warning("Empty text prompt provided")

            use_zero_order = (search_algorithm == "zero-order")
            
            # Validate inputs
            if steps < 1:
                raise ValueError("Steps must be >= 1")
            if cfg < 0:
                raise ValueError("CFG must be >= 0")
            if denoise < 0 or denoise > 1:
                raise ValueError("Denoise must be between 0 and 1")

            # Store loaded verifier models
            verifier_names = {
                "clip": loaded_clip_score_verifier,
                "image_reward": loaded_image_reward_verifier,
                "qwen_vlm_verifier": loaded_qwen_verifier
            }

            active_verifiers = sum(1 for v in verifier_names.values() if v is not None)
            if active_verifiers == 0:
                raise ValueError("No verifiers provided - at least one verifier is required")
            
            logger.info(f"Using {active_verifiers} active verifiers")

            if use_zero_order:
                result = self._execute_zero_order(
                    model, vae, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise, text_prompt_to_compare,
                    verifier_names, search_rounds, num_neighbors, lambda_threshold, view_top_k
                )
            else:
                result = self._execute_random_search(
                    model, vae, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise, text_prompt_to_compare,
                    verifier_names, search_rounds, view_top_k
                )

            logger.info("InferenceTimeScaler execution completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in InferenceTimeScaler: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return a blank image and error message
            error_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            error_json = json.dumps({"error": str(e)})
            return (error_tensor, error_tensor, error_json)

    def _execute_zero_order(self, model, vae, seed, steps, cfg, sampler_name, scheduler,
                           positive, negative, latent_image, denoise, text_prompt_to_compare,
                           verifier_names, search_rounds, num_neighbors, lambda_threshold, view_top_k):
        """
        Execute zero-order optimization search algorithm
        """
        logger.info(f"Starting zero-order search with {search_rounds} rounds")
        batch_inds = latent_image.get("batch_index", None)
        pivot_noise = comfy.sample.prepare_noise(latent_image["samples"], seed, batch_inds)
        pivot_included_in_pool = False

        # Initialize a pool to accumulate (candidate_tensor, raw_scores) pairs
        candidate_pool = []

        for round_idx in range(search_rounds):
            # Generate neighbors around the pivot noise
            neighbors = generate_neighbors(pivot_noise, threshold=lambda_threshold, num_neighbors=num_neighbors)

            # Prepend the pivot as the "zeroth" neighbor
            pivot_expanded = pivot_noise.unsqueeze(1)  # shape: [B,1,C,H,W]
            neighbors = torch.cat([pivot_expanded, neighbors], dim=1)  # shape: [B, num_neighbors+1, C,H,W]

            candidate_noises = []
            candidate_decoded_tensors = []

            for i in range(num_neighbors + 1):
                candidate_noises.append(neighbors[:, i])  # each candidate is of shape [B, C, H, W]

            # Decode all candidate noises for this round
            for noise_candidate in candidate_noises:
                samples, = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler, 
                    positive, negative, latent_image, denoise,
                    override_noise=noise_candidate
                )
                decoded = vae.decode(samples["samples"])
                if decoded.ndim == 5:
                    b, c, h, w, d = decoded.shape
                    decoded = decoded.view(-1, h, w, d)
                candidate_decoded_tensors.append(decoded[0:1])

            # score the candidates
            raw_scores = score_candidates(candidate_decoded_tensors, text_prompt_to_compare, verifier_names)
            ranked_candidates = rank_from_raw_scores(raw_scores)
            best_idx = ranked_candidates[0]["index"]

            for i, (tensor_img, candidate_raw_scores) in enumerate(zip(candidate_decoded_tensors, raw_scores)):
                if i == 0:  # pivot candidate
                    if not pivot_included_in_pool:
                        candidate_pool.append((tensor_img, candidate_raw_scores))
                        pivot_included_in_pool = True
                else:
                    candidate_pool.append((tensor_img, candidate_raw_scores))

            # Rejecting a round if the pivot is best
            if best_idx == 0:
                # The original pivot outperformed all neighbors so we reject this round
                pivot_noise = comfy.sample.prepare_noise(latent_image["samples"], seed + round_idx + 1000, batch_inds)
                pivot_included_in_pool = False
            else:
                # Update the pivot to be the best candidate's noise from this round
                pivot_noise = candidate_noises[best_idx]
                pivot_included_in_pool = True

        # Final ranking from the candidate pool 
        pool_raw_scores = [scores for (_, scores) in candidate_pool]
        ranked_pool = rank_from_raw_scores(pool_raw_scores)
        # Reorder candidate_pool by the ranking results
        candidate_pool_sorted = [candidate_pool[entry["index"]][0] for entry in ranked_pool]
        top_k_tensors = candidate_pool_sorted[:view_top_k]
        best_tensor = top_k_tensors[0]

        # Build ranking result list for JSON output
        top_k_results = ranked_pool[:view_top_k]

        # Create grid of top-k images for visualization purposes
        if len(top_k_tensors) == 0:
            grid_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            images_for_grid = [t.squeeze(0).permute(2, 0, 1) for t in top_k_tensors]
            grid = make_grid(torch.stack(images_for_grid), nrow=len(images_for_grid))
            grid_image = grid.permute(1, 2, 0).unsqueeze(0)

        results_json = {
            "results": top_k_results
        }
        scores_json_str = json.dumps(results_json, indent=2)

        return (best_tensor, grid_image, scores_json_str)

    def _execute_random_search(self, model, vae, seed, steps, cfg, sampler_name, scheduler,
                             positive, negative, latent_image, denoise, text_prompt_to_compare,
                             verifier_names, search_rounds, view_top_k):
        """
        Execute random search algorithm
        """
        logger.info(f"Starting random search with budget {search_rounds}")
        candidate_tensors = []
        for i in range(search_rounds):
            new_seed = seed + i + 1
            samples, = common_ksampler(model, new_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            decoded = vae.decode(samples["samples"])
            if decoded.ndim == 5:
                b, c, h, w, d = decoded.shape
                decoded = decoded.view(-1, h, w, d)
            single_candidate = decoded[0:1]
            candidate_tensors.append(single_candidate)

        # scoring and ranking
        raw_scores = score_candidates(candidate_tensors, text_prompt_to_compare, verifier_names)
        final_ranked_results = rank_from_raw_scores(raw_scores)
        top_k_results = final_ranked_results[:min(view_top_k, len(final_ranked_results))]
        top_k_indices = [r["index"] for r in top_k_results]
        top_k_tensors = [candidate_tensors[i] for i in top_k_indices]
        best_tensor = candidate_tensors[final_ranked_results[0]["index"]]

        # Create grid of top-k images for visualization purposes
        if len(top_k_tensors) == 0:
            grid_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            images_for_grid = [t.squeeze(0).permute(2, 0, 1) for t in top_k_tensors]
            grid = make_grid(torch.stack(images_for_grid), nrow=len(images_for_grid))
            grid_image = grid.permute(1, 2, 0).unsqueeze(0)

        results_json = {
            "results": top_k_results
        }
        scores_json_str = json.dumps(results_json, indent=2)

        return (best_tensor, grid_image, scores_json_str)


class LoadQwenVLMVerifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_verifier_id": (["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"], {
                    "default": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "tooltip": "Identifier for the Qwen VLM model."
                }),
                "device": ("STRING", {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "tooltip": "Device to load the model onto."
                }),
                "score_type": (["overall_score", "accuracy_to_prompt", "creativity_and_originality", 
                               "visual_quality_and_realism", "consistency_and_cohesion", 
                               "emotional_or_thematic_resonance"], {
                    "default": "overall_score",
                    "tooltip": "Type of score to return from the Qwen model evaluation."
                })
            }
        }

    RETURN_TYPES = ("QWN_VERIFIER",)
    RETURN_NAMES = ("qwen_verifier_instance",)
    FUNCTION = "execute"
    CATEGORY = "InferenceTimeScaling"
    DESCRIPTION = "Loads the Qwen VLM verifier model. Downloads it if necessary."

    def execute(self, qwen_verifier_id, device, score_type):
        # Construct a local comfyui checkpoint path for the model
        model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", os.path.basename(qwen_verifier_id))
        if not os.path.exists(model_checkpoint):
            snapshot_download(
                repo_id=qwen_verifier_id,
                local_dir=model_checkpoint,
                local_dir_use_symlinks=False,
            )
        verifier_instance = QwenVLMVerifier(model_checkpoint, device, score_type=score_type)
        return (verifier_instance,)


class CLIPScoreVerifier:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def score(self, text_prompt, pil_img):
        # Prepare inputs and move to the correct device
        inputs = self.processor(text=[text_prompt], images=pil_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Compute features
        text_emb = self.model.get_text_features(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_emb = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        # Normalize and compute cosine similarity
        text_emb = F.normalize(text_emb, dim=-1)
        image_emb = F.normalize(image_emb, dim=-1)
        return F.cosine_similarity(text_emb, image_emb).item()


class LoadCLIPScoreVerifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_verifier_id": (["openai/clip-vit-base-patch32", 
                                     "openai/clip-vit-large-patch14",
                                     "openai/clip-vit-base-patch16", 
                                     "openai/clip-vit-large-patch14-336"], {
                    "default": "openai/clip-vit-base-patch32",
                    "tooltip": "Identifier for the CLIP model."
                }),
                "device": ("STRING", {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "tooltip": "Device to load the model onto."
                })
            }
        }

    RETURN_TYPES = ("CS_VERIFIER",)
    RETURN_NAMES = ("clip_verifier_instance",)
    FUNCTION = "execute"
    CATEGORY = "InferenceTimeScaling"
    DESCRIPTION = "Loads the CLIP verifier model and processor."

    def execute(self, clip_verifier_id, device):
        clip_model = CLIPModel.from_pretrained(clip_verifier_id)
        clip_processor = CLIPProcessor.from_pretrained(clip_verifier_id)
        clip_model.to(device)
        verifier_instance = CLIPScoreVerifier(clip_model, clip_processor, device)
        return (verifier_instance,)


class ImageRewardVerifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def score(self, text_prompt, pil_img):
        return self.model.score(text_prompt, pil_img)


class LoadImageRewardVerifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ir_verifier_id": (["ImageReward-v1.0"], {
                    "default": "ImageReward-v1.0",
                    "tooltip": "Identifier for the ImageReward model."
                }),
                "device": ("STRING", {
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "tooltip": "Device to load the model onto."
                })
            }
        }

    RETURN_TYPES = ("IR_VERIFIER",)
    RETURN_NAMES = ("image_reward_verifier_instance",)
    FUNCTION = "execute"
    CATEGORY = "InferenceTimeScaling"
    DESCRIPTION = "Loads the ImageReward verifier model."

    def execute(self, ir_verifier_id, device):
        ir_model = reward.load(ir_verifier_id)
        try:
            ir_model.to(device)
        except Exception:
            # Some ImageReward models may not support .to(device) directly
            pass
        verifier_instance = ImageRewardVerifier(ir_model, device)
        return (verifier_instance,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "InferenceTimeScaler": InferenceTimeScaler,
    "LoadQwenVLMVerifier": LoadQwenVLMVerifier,
    "LoadCLIPScoreVerifier": LoadCLIPScoreVerifier,
    "LoadImageRewardVerifier": LoadImageRewardVerifier
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InferenceTimeScaler": "Inference Time Scaler",
    "LoadQwenVLMVerifier": "Load Qwen VLM Verifier",
    "LoadCLIPScoreVerifier": "Load CLIPScore Verifier",
    "LoadImageRewardVerifier": "Load ImageReward Verifier"
}
