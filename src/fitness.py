"""
Fitness evaluation for Experiment 1: CLIP + Aesthetic scoring
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import Optional


class FitnessEvaluator:
    """
    Fitness evaluator using CLIP and simplified aesthetic scoring

    Fitness = w1 * CLIP_score + w2 * aesthetic_score
    """

    def __init__(
        self,
        clip_weight: float = 0.6,
        aesthetic_weight: float = 0.4,
        device: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        """
        Initialize fitness evaluator

        Args:
            clip_weight: Weight for CLIP score (0-1)
            aesthetic_weight: Weight for aesthetic score (0-1)
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            clip_model_name: CLIP model identifier
        """
        self.clip_weight = clip_weight
        self.aesthetic_weight = aesthetic_weight

        # Normalize weights to sum to 1
        total = clip_weight + aesthetic_weight
        self.clip_weight = clip_weight / total
        self.aesthetic_weight = aesthetic_weight / total

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing CLIP model on {self.device}...")

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.clip_model.eval()

        print("CLIP model loaded successfully")

    def evaluate(
        self,
        image: Image.Image,
        text: str,
        verbose: bool = False
    ) -> float:
        """
        Evaluate fitness of an image-text pair

        Args:
            image: PIL Image
            text: Text prompt
            verbose: If True, print component scores

        Returns:
            Fitness score (0-1, higher is better)
        """
        clip_score = self._clip_score(image, text)
        aesthetic_score = self._aesthetic_score(image)

        fitness = (
            self.clip_weight * clip_score +
            self.aesthetic_weight * aesthetic_score
        )

        if verbose:
            print(f"CLIP score: {clip_score:.4f}")
            print(f"Aesthetic score: {aesthetic_score:.4f}")
            print(f"Weighted fitness: {fitness:.4f}")

        return fitness

    def _clip_score(self, image: Image.Image, text: str) -> float:
        """
        Calculate CLIP similarity between image and text

        Args:
            image: PIL Image
            text: Text prompt

        Returns:
            CLIP similarity score (0-1)
        """
        with torch.no_grad():
            # Process image
            image_inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

            # Process text
            text_inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Get embeddings
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)

            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity (-1 to 1)
            cosine_similarity = (image_features @ text_features.T).item()

            # Convert to 0-1 range: (cosine + 1) / 2
            similarity = (cosine_similarity + 1.0) / 2.0

        return similarity

    def _aesthetic_score(self, image: Image.Image) -> float:
        """
        Calculate aesthetic score combining CLIP feature quality and image statistics.

        Uses multiple signals:
        1. CLIP feature variance (higher = more distinctive/interesting features)
        2. Color saturation (moderate saturation preferred)
        3. Contrast (moderate-high contrast preferred)

        Args:
            image: PIL Image

        Returns:
            Aesthetic score (0-1)
        """
        # Component 1: CLIP feature-based score
        clip_aesthetic = self._clip_feature_aesthetic(image)

        # Component 2: Image statistics-based score
        image_aesthetic = self._image_statistics_aesthetic(image)

        # Combine scores (weighted average)
        aesthetic_score = 0.6 * clip_aesthetic + 0.4 * image_aesthetic

        return aesthetic_score

    def _clip_feature_aesthetic(self, image: Image.Image) -> float:
        """
        Calculate aesthetic score from CLIP feature statistics.

        Higher feature variance suggests more distinctive visual content.
        """
        with torch.no_grad():
            # Process image
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get image features (before normalization)
            image_features = self.clip_model.get_image_features(**inputs)

            # Calculate feature statistics
            feature_std = torch.std(image_features, dim=-1).item()
            feature_mean = torch.mean(torch.abs(image_features), dim=-1).item()

            # Higher variance = more distinctive features (normalize to ~0-1)
            # Typical std is around 0.3-0.6, mean abs is around 0.2-0.5
            variance_score = min(feature_std / 0.5, 1.0)
            activation_score = min(feature_mean / 0.4, 1.0)

            clip_aesthetic = 0.6 * variance_score + 0.4 * activation_score

        return clip_aesthetic

    def _image_statistics_aesthetic(self, image: Image.Image) -> float:
        """
        Calculate aesthetic score from basic image statistics.

        Considers saturation, contrast, and brightness balance.
        """
        # Convert to numpy array
        img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0

        # Calculate color saturation
        # Higher saturation (within reason) often looks better
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)
        mean_saturation = np.mean(saturation)
        # Optimal saturation around 0.3-0.5, penalize extremes
        saturation_score = 1.0 - abs(mean_saturation - 0.4) * 2
        saturation_score = max(0.0, min(1.0, saturation_score))

        # Calculate contrast (standard deviation of luminance)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        contrast = np.std(luminance)
        # Good contrast is around 0.15-0.3
        contrast_score = min(contrast / 0.2, 1.0)

        # Calculate brightness balance (penalize too dark or too bright)
        mean_brightness = np.mean(luminance)
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
        brightness_score = max(0.0, min(1.0, brightness_score))

        # Combine image statistics
        image_aesthetic = (
            0.35 * saturation_score +
            0.40 * contrast_score +
            0.25 * brightness_score
        )

        return image_aesthetic


class AdaptiveFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator with adaptive weights based on generation number

    Early generations: Focus more on CLIP (semantic alignment)
    Later generations: Focus more on aesthetics (visual quality)
    """

    def __init__(
        self,
        initial_clip_weight: float = 0.8,
        final_clip_weight: float = 0.4,
        max_generations: int = 20,
        device: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        """
        Initialize adaptive fitness evaluator

        Args:
            initial_clip_weight: CLIP weight at generation 0
            final_clip_weight: CLIP weight at final generation
            max_generations: Maximum number of generations
            device: Device to run models on
            clip_model_name: CLIP model identifier
        """
        # Start with initial weights
        super().__init__(
            clip_weight=initial_clip_weight,
            aesthetic_weight=1.0 - initial_clip_weight,
            device=device,
            clip_model_name=clip_model_name
        )

        self.initial_clip_weight = initial_clip_weight
        self.final_clip_weight = final_clip_weight
        self.max_generations = max_generations
        self.current_generation = 0

    def update_generation(self, generation: int):
        """
        Update weights based on current generation

        Args:
            generation: Current generation number
        """
        self.current_generation = generation

        # Linear interpolation between initial and final weights
        progress = min(generation / self.max_generations, 1.0)
        self.clip_weight = (
            self.initial_clip_weight +
            progress * (self.final_clip_weight - self.initial_clip_weight)
        )
        self.aesthetic_weight = 1.0 - self.clip_weight

    def evaluate(
        self,
        image: Image.Image,
        text: str,
        verbose: bool = False
    ) -> float:
        """
        Evaluate fitness with adaptive weights

        Args:
            image: PIL Image
            text: Text prompt
            verbose: If True, print component scores and current weights

        Returns:
            Fitness score
        """
        if verbose:
            print(f"Generation {self.current_generation}: "
                  f"CLIP weight={self.clip_weight:.2f}, "
                  f"Aesthetic weight={self.aesthetic_weight:.2f}")

        return super().evaluate(image, text, verbose)
