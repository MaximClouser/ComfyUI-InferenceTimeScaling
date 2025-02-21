import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import latent_preview
import json
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from transformers import CLIPProcessor, CLIPModel
import ImageReward as reward


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


def compute_similarity(clip_model, clip_processor, text, image) -> float:
    # Process inputs: 'text' is a string, and 'image' should be a PIL image
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get text and image embeddings.
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        image_embeddings = clip_model.get_image_features(
            pixel_values=inputs['pixel_values']
        )
    
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    
    similarity = F.cosine_similarity(text_embeddings, image_embeddings)
    # rounded_similarity = round(similarity.item(), 4)
    rounded_similarity = similarity.item()
    return rounded_similarity


def compute_combined_verifier_score(candidate_image, text_prompt, verifiers, weights):
    """
    verifiers: a dict with keys like 'clip' and 'image_reward' and values holding the corresponding model(s)/processor(s)
    weights: a dict with the same keys, providing a float weight for each verifier
    Returns a combined score and a dict of individual scores.
    """
    scores = {}

    if 'clip' in verifiers and verifiers['clip'] is not None:
        clip_model, clip_processor = verifiers['clip']
        clip_score = compute_similarity(clip_model, clip_processor, text_prompt, candidate_image)
        scores['clip'] = clip_score

    if 'image_reward' in verifiers and verifiers['image_reward'] is not None:
        image_reward_model = verifiers['image_reward']
        image_reward_score = image_reward_model.score(text_prompt, candidate_image)
        scores['image_reward'] = image_reward_score

    combined_score = 0.0
    for key, score in scores.items():
        # TODO Implement algo from paper to combine scores
        combined_score += weights.get(key, 1.0) * score
        combined_score = round(combined_score, 4)
    return combined_score, scores


class MaximKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt; however, too high values may negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampling algorithm used. This can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied; lower values maintain more of the structure of the initial image."}),
                "text_prompt_to_compare": ("STRING", {"default": "", "multiline": True, "tooltip": "A text prompt used for comparing against generated images."}),
                "search_budget": ("INT", {"default": 5, "min": 1, "max": 10000, "tooltip": "The number of random noise samples to generate (i.e., the search budget)."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent image."}),
                "top_k": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "The top k image/score pairs to return."}),
            },
            "optional": {
                "clip_score_verifier": ("STRING", {"default": "openai/clip-vit-base-patch32", "tooltip": "Optional Hugging Face CLIP model identifier used for verifying the generated image quality."}),
                "clip_score_weight": ("FLOAT", {"default": 1, "min": 0, "max": 1, "tooltip": "The weight of the clip verifier to apply "}),
                "image_reward_verifier": ("STRING", {"default": "ImageReward-v1.0", "tooltip": "Optional Image Reward model used for verifying the generated image quality."}),
                "image_reward_weight": ("FLOAT", {"default": 1, "min": 0, "max": 1, "tooltip": "The weight of the iamge rewqrd to apply "})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Best Image", "Top-k Grid", "Top-k Score(s)")

    OUTPUT_TOOLTIPS = ("A grid of the top k images from the random search, and a JSON string with their verifier scores.",)
    FUNCTION = "execute"

    CATEGORY = "maxtest/sampling"
    DESCRIPTION = ("Uses the provided model, conditioning, and VAE to denoise the latent image. "
                   "Performs random search over noise candidates by sampling different noises, decodes each candidate image, "
                   "evaluates its quality using a verifier (if provided), and returns the top k candidates based on the verifier score.")


    def execute(self, 
                model, 
                vae, 
                seed, 
                steps, 
                cfg, 
                sampler_name, 
                scheduler, 
                positive, 
                negative, 
                latent_image, 
                denoise, 
                text_prompt_to_compare, 
                search_budget,
                top_k,
                clip_score_verifier=None,
                image_reward_verifier=None,
                clip_score_weight=1.0,
                image_reward_weight=1.0
    ):
        # TODO Move to innit
        device = "cuda" if torch.cuda.is_available() else "cpu"

        candidate_results = []

        verifiers = {}
        weights = {}
        
        if clip_score_verifier:
            clip_model = CLIPModel.from_pretrained(clip_score_verifier)
            clip_processor = CLIPProcessor.from_pretrained(clip_score_verifier)
            clip_model.to(device)
            verifiers['clip'] = (clip_model, clip_processor)
            weights['clip'] = clip_score_weight

        if image_reward_verifier:
            image_reward_model = reward.load(image_reward_verifier)
            verifiers['image_reward'] = image_reward_model
            weights['image_reward'] = image_reward_weight

        for i in range(search_budget):
            new_seed = seed + i + 1
            samples, = common_ksampler(model, new_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            
            candidate_images = vae.decode(samples["samples"])
            if len(candidate_images.shape) == 5:
                candidate_images = candidate_images.reshape(-1, candidate_images.shape[-3], candidate_images.shape[-2], candidate_images.shape[-1])
            
            candidate_image = candidate_images[0:1]  # Using the first image of the batch.

            # Convert from tensor to PIL Image
            # Remove the batch dimension: [1, H, W, 3] -> [H, W, 3]
            image_tensor = candidate_image.squeeze(0)
            # Permute to [C, H, W]
            image_tensor = image_tensor.permute(2, 0, 1)
            # Convert to PIL Image
            candidate_pil_image = ToPILImage()(image_tensor)

            # Score the candidate
            combined_score, individual_scores = compute_combined_verifier_score(candidate_pil_image, text_prompt_to_compare, verifiers, weights)
            
            candidate_results.append((candidate_image, combined_score))
        
        candidate_results.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidate_results[:min(top_k, len(candidate_results))]
        
        images_list = [img for (img, score) in top_candidates]
        scores_list = [score for (img, score) in top_candidates]
        
        # Prepare the images for grid display.
        images_for_grid = [img.squeeze(0).permute(2, 0, 1) for img in images_list]
        grid_image = make_grid(torch.stack(images_for_grid), nrow=top_k)
        grid_image = grid_image.permute(1, 2, 0).unsqueeze(0)
        
        scores_json = json.dumps({"results": [{"score": score} for score in scores_list]})
        best_candidate = images_list[0]
        return (best_candidate, grid_image, scores_json)






# Node registration
NODE_CLASS_MAPPINGS = {
    "MaximKSampler": MaximKSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaximKSampler": "MaximKSampler"
}


