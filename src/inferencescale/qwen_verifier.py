import logging
import torch
import outlines
from PIL import Image
import gc

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

from outlines.models.transformers_vision import transformers_vision
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Prompt from: Ma, Nanye, et al. "Inference-time scaling for diffusion models beyond scaling denoising steps." arXiv preprint arXiv:2501.09732 (2025).
SYSTEM_PROMPT = """
    You are a multimodal large-language model tasked with evaluating images
    generated by a text-to-image model. Your goal is to assess each generated
    image based on specific aspects and provide a detailed critique, along with
    a scoring system. The final output should be formatted as a JSON object
    containing individual scores for each aspect and an overall score. Below
    is a comprehensive guide to follow in your evaluation process:
    1. Key Evaluation Aspects and Scoring Criteria:
    For each aspect, provide a score from 0 to 10, where 0 represents poor
    performance and 10 represents excellent performance. For each score, include
    a short explanation or justification (1-2 sentences) explaining why that
    score was given. The aspects to evaluate are as follows:
    a) Accuracy to Prompt
    Assess how well the image matches the description given in the prompt.
    Consider whether all requested elements are present and if the scene,
    objects, and setting align accurately with the text. Score: 0 (no
    alignment) to 10 (perfect match to prompt).
    b) Creativity and Originality
    Evaluate the uniqueness and creativity of the generated image. Does the
    model present an imaginative or aesthetically engaging interpretation of the
    prompt? Is there any evidence of creativity beyond a literal interpretation?
    Score: 0 (lacks creativity) to 10 (highly creative and original).
    c) Visual Quality and Realism
    Assess the overall visual quality, including resolution, detail, and realism.
    Look for coherence in lighting, shading, and perspective. Even if the image
    is stylized or abstract, judge whether the visual elements are well-rendered
    and visually appealing. Score: 0 (poor quality) to 10 (high-quality and
    realistic).
    d) Consistency and Cohesion
    Check for internal consistency within the image. Are all elements cohesive
    and aligned with the prompt? For instance, does the perspective make sense,
    and do objects fit naturally within the scene without visual anomalies?
    Score: 0 (inconsistent) to 10 (fully cohesive and consistent).
    e) Emotional or Thematic Resonance
    Evaluate how well the image evokes the intended emotional or thematic tone of
    the prompt. For example, if the prompt is meant to be serene, does the image
    convey calmness? If it's adventurous, does it evoke excitement? Score: 0
    (no resonance) to 10 (strong resonance with the prompt's theme).
    2. Overall Score
    After scoring each aspect individually, provide an overall score,
    representing the model's general performance on this image. This should be
    a weighted average based on the importance of each aspect to the prompt or an
    average of all aspects.
"""


class Score(BaseModel):
    explanation: str
    score: float


class Grading(BaseModel):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class QwenVLMVerifier():

    def __init__(self, model_name, device='cpu'):
        logger.info(f"Initializing QwenVLMVerifier with model {model_name} on device {device}")
        self.model_name = model_name
        self.device = device
        self.dtype = torch.float16 if "cuda" in self.device else torch.float16
        self.load_model()


    def load_model(self):
        logger.info("Starting model loading process")
        try:
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28

            logger.debug(f"Loading model from {self.model_name}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=self.dtype, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully")

            logger.debug("Loading processor")
            processor = AutoProcessor.from_pretrained(
                self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )
            logger.info("Processor loaded successfully")

            logger.debug("Initializing transformers vision")
            self.qwen_model = transformers_vision(
                self.model_name,
                model_class=model.__class__,
                device=self.device,
                model_kwargs={"torch_dtype": self.dtype},
                processor_class=processor.__class__,
            )
            logger.info("Transformers vision initialized")

            logger.debug("Setting up structured generator")
            self.structured_qwen_generator = outlines.generate.json(self.qwen_model, Grading)
            logger.info("Structured generator setup complete")

            del model 
            del processor
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memory cleanup completed")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise


    # def to_device(self, device: str):
    #     """
    #     Moves the underlying model to the specified device.
    #     Also updates the instance's device and dtype accordingly.
    #     """
    #     self.device = device
    #     self.dtype = torch.float16 if "cuda" in device else torch.float16
    #     # Move the underlying model to the target device.
    #     self.qwen_model.model.to(device)
    #     logger.info(f"Moved model to {device}")


    def query_model(self, image, prompt: str, max_tokens: int = None, seed: int = 42) -> dict:
        # TODO Batch processing
        logger.info(f"Querying Qwen model...")
        try:
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.qwen_model.processor.apply_chat_template(conversation, add_generation_prompt=True)
            logger.debug("Generated chat template")
            
            outputs = self.structured_qwen_generator([text_prompt], [[image]], max_tokens=max_tokens, seed=seed)
            logger.info("Successfully generated response")
            return outputs[0].dict()
        except Exception as e:
            logger.error(f"Error during model query: {str(e)}", exc_info=True)
            raise


    def get_overall_score(self, image, prompt: str, max_tokens: int = None, seed: int = 42) -> float:
        # TODO Extend to handle any score key (instead of just overall score) as input
        logger.info("Getting overall score")
        try:
            outputs = self.query_model(image, prompt, max_tokens, seed)
            overall_score = outputs["overall_score"]["score"]

            if overall_score:
                logger.debug(f"Overall score calculated: {overall_score}")
                return float(overall_score)

            logger.warning("Overall score not found in model output")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting overall score: {str(e)}", exc_info=True)
            return 0.0


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    model = QwenVLMVerifier(model_name=model_name, device=device)
    # model.load_model()
    
    image_path = "596F6DF4-2856-436E-A981-649ABFB15F1B.jpeg"
    image = Image.open(image_path).convert("RGB")

    test_prompt = "A red bird and a fish."
    
    response = model.query_model(image, test_prompt)
    print("Model Response:", response)

    aspect_keys = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance"
    ]
    
    scores = []
    for key in aspect_keys:
        if key in response and "score" in response[key]:
            scores.append(response[key]["score"])
    
    if scores:
        average_score = sum(scores) / len(scores)
        print("Average Score:", average_score)
    else:
        print("No scores found to average.")

    # model.to_device('cpu')
    # model.to_device('cuda')

    response = model.query_model(image, test_prompt)
    print("Model Response:", response)

    overall_score = model.get_overall_score(image, test_prompt)
    print(f"Overall score: {overall_score}")
