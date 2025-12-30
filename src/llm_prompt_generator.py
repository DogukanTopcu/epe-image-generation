"""
LLM-based prompt generator using Gemini for template analysis
"""

import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Optional
from PIL import Image
from abc import ABC, abstractmethod
import random


class PromptGenerator(ABC):
    """Base class for prompt generators"""

    @abstractmethod
    def analyze_reference_image(
        self,
        reference: Image.Image,
        user_subject: str,
        num_variations: int = 3
    ) -> List[Dict[str, List[str]]]:
        """
        Analyze reference image and generate structured prompts

        Args:
            reference: Reference PIL Image
            user_subject: User's desired subject
            num_variations: Number of prompt variations to generate

        Returns:
            List of structured prompt dictionaries
        """
        pass

    @abstractmethod
    def generate_seed_prompts(
        self,
        reference: Image.Image,
        user_subject: str,
        population_size: int
    ) -> List[Dict[str, List[str]]]:
        """
        Generate seed prompts for initial population

        Args:
            reference: Reference PIL Image
            user_subject: User's desired subject
            population_size: Target population size

        Returns:
            List of structured prompt dictionaries
        """
        pass


class GeminiPromptGenerator(PromptGenerator):
    """Gemini-based prompt generator using Vertex AI"""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize Gemini prompt generator

        Args:
            project_id: Google Cloud project ID (or from env)
            location: Google Cloud location (or from env)
            model_name: Gemini model name
        """
        try:
            from google.cloud import aiplatform
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required for GeminiPromptGenerator. "
                "Install with: pip install google-cloud-aiplatform"
            )

        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set in environment or passed as argument"
            )

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)

        # Initialize Gemini model
        self.model = GenerativeModel(model_name)
        self.Part = Part

        print(f"Initialized Gemini ({model_name}) on project {self.project_id}")

    def analyze_reference_image(
        self,
        reference: Image.Image,
        user_subject: str,
        num_variations: int = 3
    ) -> List[Dict[str, List[str]]]:
        """
        Analyze reference image using Gemini and generate structured prompts

        Args:
            reference: Reference PIL Image
            user_subject: User's desired subject
            num_variations: Number of prompt variations to generate

        Returns:
            List of structured prompt dictionaries
        """
        # Convert image to base64
        image_bytes = self._image_to_bytes(reference)

        # Create prompt for Gemini
        prompt = self._create_analysis_prompt(user_subject, num_variations)

        # Call Gemini
        try:
            response = self.model.generate_content([
                self.Part.from_data(image_bytes, mime_type="image/jpeg"),
                prompt
            ])

            # Parse JSON response
            response_text = response.text.strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            prompts = json.loads(response_text)

            # Validate structure
            if not isinstance(prompts, list):
                raise ValueError("Response is not a list")

            return prompts

        except Exception as e:
            print(f"Error calling Gemini: {e}")
            print(f"Response text: {response_text if 'response_text' in locals() else 'N/A'}")
            # Fallback to dummy prompts
            return self._generate_fallback_prompts(num_variations)

    def generate_seed_prompts(
        self,
        reference: Image.Image,
        user_subject: str,
        population_size: int
    ) -> List[Dict[str, List[str]]]:
        """
        Generate seed prompts for initial population

        Calls analyze_reference_image and creates variations to fill population

        Args:
            reference: Reference PIL Image
            user_subject: User's desired subject
            population_size: Target population size

        Returns:
            List of structured prompt dictionaries
        """
        # Generate base variations (5-8 prompts)
        num_base_variations = min(8, max(5, population_size // 2))
        base_prompts = self.analyze_reference_image(
            reference,
            user_subject,
            num_variations=num_base_variations
        )

        # If we need more prompts, create variations of existing ones
        seed_prompts = base_prompts.copy()

        while len(seed_prompts) < population_size:
            # Pick a random base prompt and vary it slightly
            base = random.choice(base_prompts)
            varied = self._create_variation(base)
            seed_prompts.append(varied)

        return seed_prompts[:population_size]

    def _create_analysis_prompt(self, user_subject: str, num_variations: int) -> str:
        """Create the prompt for Gemini to analyze the image"""
        return f"""Analyze this reference image and generate {num_variations} structured prompts.

User's subject (MUST include): {user_subject}

Task:
1. Identify composition (camera angle, framing, layout)
2. Identify lighting (type, direction, mood)
3. Identify style (artistic approach, rendering)
4. Generate {num_variations} variations that match the structure but with the user's subject

Output JSON array:
[
  {{
    "composition": ["wide angle", "centered", ...],
    "lighting": ["golden hour", "soft light", ...],
    "style": ["photorealistic", "cinematic", ...],
    "quality": ["8k", "detailed", ...],
    "negative": ["blurry", "distorted", ...]
  }},
  ...
]

Rules:
- Each block should have 1-3 items
- Use descriptive, specific terms
- Ensure variations are diverse
- Output ONLY valid JSON, no other text

Output the JSON array now:"""

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to JPEG bytes"""
        buffer = BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()

    def _generate_fallback_prompts(self, num_variations: int) -> List[Dict[str, List[str]]]:
        """Generate fallback prompts if Gemini fails"""
        print("Using fallback prompts")
        from src.utils import create_block_vocabularies

        vocabs = create_block_vocabularies()

        prompts = []
        for _ in range(num_variations):
            prompt = {
                "composition": random.sample(vocabs["composition"], 2),
                "lighting": random.sample(vocabs["lighting"], 2),
                "style": random.sample(vocabs["style"], 2),
                "quality": random.sample(vocabs["quality"], 2),
                "negative": random.sample(vocabs["negative"], 3)
            }
            prompts.append(prompt)

        return prompts

    def _create_variation(self, base_prompt: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create a variation of an existing prompt"""
        from src.utils import create_block_vocabularies

        vocabs = create_block_vocabularies()
        varied = {}

        for block_name, values in base_prompt.items():
            # Keep most values, change 1-2
            new_values = values.copy()

            if len(new_values) > 0 and random.random() < 0.5:
                # Replace one value
                vocab = vocabs.get(block_name, [])
                if vocab:
                    available = [v for v in vocab if v not in new_values]
                    if available:
                        idx = random.randrange(len(new_values))
                        new_values[idx] = random.choice(available)

            varied[block_name] = new_values

        return varied


class DummyPromptGenerator(PromptGenerator):
    """Dummy prompt generator for testing without Gemini"""

    def __init__(self):
        print("Initialized DummyPromptGenerator (no LLM calls)")

    def analyze_reference_image(
        self,
        reference: Image.Image,
        user_subject: str,
        num_variations: int = 3
    ) -> List[Dict[str, List[str]]]:
        """Generate random prompts without analyzing image"""
        from src.utils import create_block_vocabularies

        vocabs = create_block_vocabularies()

        prompts = []
        for i in range(num_variations):
            # Generate diverse random prompts
            prompt = {
                "composition": random.sample(vocabs["composition"], random.randint(1, 3)),
                "lighting": random.sample(vocabs["lighting"], random.randint(1, 3)),
                "style": random.sample(vocabs["style"], random.randint(1, 3)),
                "quality": random.sample(vocabs["quality"], random.randint(2, 3)),
                "negative": random.sample(vocabs["negative"], random.randint(2, 4))
            }
            prompts.append(prompt)

        return prompts

    def generate_seed_prompts(
        self,
        reference: Image.Image,
        user_subject: str,
        population_size: int
    ) -> List[Dict[str, List[str]]]:
        """Generate random seed prompts"""
        return self.analyze_reference_image(reference, user_subject, population_size)


def get_prompt_generator(use_llm: bool = True, **kwargs) -> PromptGenerator:
    """
    Factory function to get a prompt generator

    Args:
        use_llm: If True, use Gemini; otherwise use dummy generator
        **kwargs: Additional arguments for generator

    Returns:
        PromptGenerator instance
    """
    if use_llm:
        try:
            return GeminiPromptGenerator(**kwargs)
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            print("Falling back to DummyPromptGenerator")
            return DummyPromptGenerator()
    else:
        return DummyPromptGenerator()
