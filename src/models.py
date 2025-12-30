"""
Text-to-image model wrappers for Fal AI API
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import requests
from io import BytesIO
from PIL import Image
import fal_client
from dotenv import load_dotenv

load_dotenv()


class FalAIClient(ABC):
    """Base class for Fal AI model clients"""

    def __init__(self):
        """Initialize the Fal AI client with API key"""
        self.api_key = os.getenv("FAL_KEY")
        if not self.api_key:
            raise ValueError("FAL_KEY not found in environment variables")

        # Set the API key for fal_client
        os.environ["FAL_KEY"] = self.api_key

    def _download_image(self, url: str) -> Image.Image:
        """
        Download an image from a URL

        Args:
            url: Image URL

        Returns:
            PIL Image object
        """
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate an image from a text prompt

        Args:
            prompt: Text prompt
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        pass


class FluxSchnellModel(FalAIClient):
    """Flux-1 Schnell model (fast, 4 steps)"""

    def __init__(self):
        super().__init__()
        self.model_id = "fal-ai/flux/schnell"

    def generate(
        self,
        prompt: str,
        image_size: str = "landscape_4_3",
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate an image using Flux-1 Schnell

        Args:
            prompt: Text prompt
            image_size: Image size preset (landscape_4_3, square, portrait_4_3, etc.)
            num_inference_steps: Number of denoising steps (default 4)
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale (optional)

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
        }

        if seed is not None:
            arguments["seed"] = seed
        if guidance_scale is not None:
            arguments["guidance_scale"] = guidance_scale

        # Add any additional kwargs
        arguments.update(kwargs)

        result = fal_client.subscribe(
            self.model_id,
            arguments=arguments
        )

        image_url = result["images"][0]["url"]
        image = self._download_image(image_url)

        metadata = {
            "model": self.model_id,
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": result.get("seed"),
            "timings": result.get("timings", {}),
        }

        return image, metadata


class QwenImageModel(FalAIClient):
    """Qwen Image model (slower, supports negative prompts)"""

    def __init__(self):
        super().__init__()
        self.model_id = "fal-ai/qwen-image"

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image_size: str = "landscape_4_3",
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate an image using Qwen Image

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (things to avoid)
            image_size: Image size preset
            num_inference_steps: Number of denoising steps (default 30)
            guidance_scale: Classifier-free guidance scale (default 3.5)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt
        if seed is not None:
            arguments["seed"] = seed

        # Add any additional kwargs
        arguments.update(kwargs)

        result = fal_client.subscribe(
            self.model_id,
            arguments=arguments
        )

        image_url = result["images"][0]["url"]
        image = self._download_image(image_url)

        metadata = {
            "model": self.model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": result.get("seed"),
            "timings": result.get("timings", {}),
        }

        return image, metadata


def get_model(model_name: str) -> FalAIClient:
    """
    Factory function to get a model instance by name

    Args:
        model_name: Model name ('flux-schnell' or 'qwen-image')

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not recognized
    """
    models = {
        "flux-schnell": FluxSchnellModel,
        "qwen-image": QwenImageModel,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )

    return models[model_name]()
