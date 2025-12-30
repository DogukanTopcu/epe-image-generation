"""
Fitness evaluation for Experiment 2: CLIP + LPIPS (template matching)
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import lpips
import torchvision.transforms as transforms
from typing import Optional


class TemplateFitnessEvaluator:
    """
    Fitness evaluator using CLIP (content) + LPIPS (structure)

    Fitness = w1 * CLIP_score + w2 * LPIPS_similarity
    """

    def __init__(
        self,
        reference_image: Image.Image,
        clip_weight: float = 0.5,
        lpips_weight: float = 0.5,
        device: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        lpips_net: str = "vgg"
    ):
        """
        Initialize template fitness evaluator

        Args:
            reference_image: Reference image for structure matching
            clip_weight: Weight for CLIP score (content alignment)
            lpips_weight: Weight for LPIPS similarity (structure matching)
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            clip_model_name: CLIP model identifier
            lpips_net: LPIPS network ('vgg', 'alex', or 'squeeze')
        """
        self.clip_weight = clip_weight
        self.lpips_weight = lpips_weight

        # Normalize weights to sum to 1
        total = clip_weight + lpips_weight
        self.clip_weight = clip_weight / total
        self.lpips_weight = lpips_weight / total

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing models on {self.device}...")

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()
        print("CLIP model loaded")

        # Load LPIPS model
        self.lpips_model = lpips.LPIPS(net=lpips_net).to(self.device)
        self.lpips_model.eval()
        print("LPIPS model loaded")

        # Preprocess reference image for LPIPS
        self.reference_tensor = self._preprocess_for_lpips(reference_image)
        print("Reference image preprocessed")

    def evaluate(
        self,
        image: Image.Image,
        text: str,
        verbose: bool = False
    ) -> float:
        """
        Evaluate fitness of an image

        Args:
            image: Generated PIL Image
            text: User prompt (subject)
            verbose: If True, print component scores

        Returns:
            Fitness score (0-1, higher is better)
        """
        clip_score = self._clip_score(image, text)
        lpips_similarity = self._lpips_similarity(image)

        fitness = (
            self.clip_weight * clip_score +
            self.lpips_weight * lpips_similarity
        )

        if verbose:
            print(f"CLIP score (content): {clip_score:.4f}")
            print(f"LPIPS similarity (structure): {lpips_similarity:.4f}")
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

    def _lpips_similarity(self, image: Image.Image) -> float:
        """
        Calculate LPIPS similarity between generated and reference image

        LPIPS returns distance (lower is better), so we convert to similarity

        Args:
            image: Generated PIL Image

        Returns:
            LPIPS similarity score (0-1, higher is better)
        """
        with torch.no_grad():
            # Preprocess generated image
            gen_tensor = self._preprocess_for_lpips(image)

            # Calculate LPIPS distance
            distance = self.lpips_model(self.reference_tensor, gen_tensor).item()

            # Convert distance to similarity
            # LPIPS distance typically ranges from 0 (identical) to ~1 (very different)
            # We use 1/(1+d) to convert to similarity
            similarity = 1.0 / (1.0 + distance)

        return similarity

    def _preprocess_for_lpips(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for LPIPS model

        LPIPS expects images in [-1, 1] range with shape (1, 3, H, W)

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if necessary (handles RGBA, L, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to 256x256 (standard for LPIPS)
        image = image.resize((256, 256), Image.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)

        return tensor


class AdaptiveTemplateFitnessEvaluator(TemplateFitnessEvaluator):
    """
    Template fitness evaluator with adaptive weights

    Early generations: Focus more on structure (LPIPS)
    Later generations: Focus more on content (CLIP)
    """

    def __init__(
        self,
        reference_image: Image.Image,
        initial_clip_weight: float = 0.3,
        final_clip_weight: float = 0.6,
        max_generations: int = 20,
        device: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        lpips_net: str = "vgg"
    ):
        """
        Initialize adaptive template fitness evaluator

        Args:
            reference_image: Reference image for structure matching
            initial_clip_weight: CLIP weight at generation 0
            final_clip_weight: CLIP weight at final generation
            max_generations: Maximum number of generations
            device: Device to run models on
            clip_model_name: CLIP model identifier
            lpips_net: LPIPS network
        """
        # Start with initial weights
        super().__init__(
            reference_image=reference_image,
            clip_weight=initial_clip_weight,
            lpips_weight=1.0 - initial_clip_weight,
            device=device,
            clip_model_name=clip_model_name,
            lpips_net=lpips_net
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
        self.lpips_weight = 1.0 - self.clip_weight

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
                  f"LPIPS weight={self.lpips_weight:.2f}")

        return super().evaluate(image, text, verbose)
