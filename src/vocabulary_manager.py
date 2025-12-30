"""
Adaptive vocabulary management using LLM.
Generates, expands, and prunes vocabulary during evolution.

This is the KEY INNOVATION of this research project:
- Initial vocabulary: LLM generates 1000 domain-specific modifiers
- Adaptive expansion: Every N generations, analyze best prompts and add new terms
- Usage-based pruning: Remove unused modifiers to maintain focus
- Synonym mappings: Enable intelligent, semantic-aware mutations
"""

import os
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()


class VocabularyManager:
    """
    Manages dynamic vocabulary evolution for prompt engineering.

    Key capabilities:
    - LLM-generated initial vocabulary (1000 modifiers)
    - Adaptive expansion based on successful prompts
    - Usage tracking and pruning
    - Synonym mappings for mutation
    """

    def __init__(
        self,
        use_llm: bool = True,
        initial_size: int = 1000,
        expansion_size: int = 50,
        prune_threshold: int = 20
    ):
        """
        Initialize vocabulary manager

        Args:
            use_llm: Use LLM for vocabulary generation
            initial_size: Target size for initial vocabulary
            expansion_size: New modifiers to add per expansion
            prune_threshold: Generations before pruning unused modifiers
        """
        self.use_llm = use_llm
        self.initial_size = initial_size
        self.expansion_size = expansion_size
        self.prune_threshold = prune_threshold

        # Vocabularies by block
        self.vocabularies: Dict[str, List[str]] = {
            "composition": [],
            "lighting": [],
            "style": [],
            "quality": [],
            "negative": []
        }

        # Synonym mappings
        self.synonym_map: Dict[str, List[str]] = {}

        # Usage tracking
        self.usage_counts: Dict[str, Counter] = {
            block: Counter() for block in self.vocabularies.keys()
        }

        # Track when each modifier was added
        self.generation_added: Dict[str, Dict[str, int]] = {
            block: {} for block in self.vocabularies.keys()
        }

        # Initialize LLM if enabled
        if use_llm:
            self._init_llm()
        else:
            self.model = None
            print("VocabularyManager: LLM disabled, using static fallback")

    def _init_llm(self):
        """Initialize Vertex AI Gemini client"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT not set")

            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel("gemini-2.0-flash-exp")

            # Import Part for image handling
            from vertexai.generative_models import Part
            self.Part = Part

            print(f"VocabularyManager: Initialized Gemini (project: {project_id})")
        except Exception as e:
            print(f"VocabularyManager: Failed to initialize LLM: {e}")
            self.model = None
            self.use_llm = False

    # ==================== INITIALIZATION ====================

    def initialize_vocabulary(
        self,
        reference_image: Optional[Image.Image] = None,
        domain_description: str = "general photography"
    ) -> Dict[str, List[str]]:
        """
        Generate initial vocabulary using LLM

        Args:
            reference_image: Optional reference for domain-specific terms
            domain_description: Domain context

        Returns:
            Dictionary of vocabularies by block
        """
        print(f"\n{'='*70}")
        print(f"INITIALIZING VOCABULARY")
        print(f"{'='*70}")
        print(f"Target size: {self.initial_size} modifiers")
        print(f"Domain: {domain_description}")
        print(f"LLM enabled: {self.use_llm}")

        if not self.use_llm or self.model is None:
            print("Using minimal static vocabulary (fallback)")
            return self._fallback_vocabulary()

        try:
            # Construct prompt
            prompt_text = self._construct_initial_vocab_prompt(domain_description)

            # Call LLM
            if reference_image:
                print("Analyzing reference image...")
                image_part = self._image_to_part(reference_image)
                response = self.model.generate_content([image_part, prompt_text])
            else:
                print("Generating vocabulary...")
                response = self.model.generate_content(prompt_text)

            # Parse response
            vocab_data = self._parse_vocabulary_response(response.text)

            self.vocabularies = vocab_data["vocabularies"]
            self.synonym_map = vocab_data["synonyms"]

            # Track all as added at generation 0
            for block_name, modifiers in self.vocabularies.items():
                for modifier in modifiers:
                    self.generation_added[block_name][modifier] = 0

            self._print_vocabulary_stats("INITIAL VOCABULARY")

            return self.vocabularies

        except Exception as e:
            print(f"ERROR generating vocabulary: {e}")
            print("Falling back to static vocabulary")
            import traceback
            traceback.print_exc()
            return self._fallback_vocabulary()

    def _construct_initial_vocab_prompt(self, domain: str) -> str:
        """Construct prompt for initial vocabulary generation"""
        per_block = self.initial_size // 5

        return f"""Generate a comprehensive vocabulary for text-to-image prompt engineering.

Domain: {domain}

Generate approximately {self.initial_size} total modifiers across 5 categories:

1. composition ({per_block} terms): camera angles, framing, composition rules, depth, perspective
2. lighting ({per_block} terms): natural, artificial, direction, quality, mood, cinematography
3. style ({per_block} terms): photographic, artistic, digital, cinematic, painting styles
4. quality ({per_block} terms): resolution, detail, sharpness, professional terms
5. negative ({per_block} terms): quality issues, anatomy problems, artifacts

Include technical terms, equipment names, specific styles, and natural language descriptions.

For ~200 key modifiers, provide 3-5 synonyms for semantic mutation.

Output ONLY valid JSON (no markdown):
{{
  "vocabularies": {{
    "composition": ["term1", "term2", ...],
    "lighting": [...],
    "style": [...],
    "quality": [...],
    "negative": [...]
  }},
  "synonyms": {{
    "term1": ["synonym1", "synonym2", ...],
    ...
  }}
}}"""

    # ==================== EXPANSION ====================

    def expand_vocabulary(
        self,
        best_prompts: List[Dict[str, List[str]]],
        current_generation: int,
        fitness_scores: List[float]
    ) -> Dict[str, List[str]]:
        """
        Expand vocabulary based on successful prompts

        Args:
            best_prompts: Top-performing prompt structures
            current_generation: Current generation number
            fitness_scores: Fitness scores of best prompts

        Returns:
            New modifiers added (dict by block)
        """
        print(f"\n{'='*70}")
        print(f"VOCABULARY EXPANSION (Generation {current_generation})")
        print(f"{'='*70}")

        if not self.use_llm or self.model is None:
            print("LLM disabled, skipping expansion")
            return {}

        try:
            prompt_text = self._construct_expansion_prompt(best_prompts, fitness_scores)
            response = self.model.generate_content(prompt_text)
            new_modifiers = self._parse_expansion_response(response.text)

            # Add to vocabularies (avoid duplicates)
            added_count = 0
            actually_added = {block: [] for block in self.vocabularies.keys()}

            for block_name, modifiers in new_modifiers.items():
                for modifier in modifiers:
                    if modifier not in self.vocabularies[block_name]:
                        self.vocabularies[block_name].append(modifier)
                        self.generation_added[block_name][modifier] = current_generation
                        actually_added[block_name].append(modifier)
                        added_count += 1

            print(f"Added {added_count} new modifiers")

            for block_name, mods in actually_added.items():
                if mods:
                    print(f"  {block_name}: {', '.join(mods[:5])}{'...' if len(mods) > 5 else ''}")

            self._print_vocabulary_stats("AFTER EXPANSION")

            return actually_added

        except Exception as e:
            print(f"ERROR during expansion: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _construct_expansion_prompt(self, best_prompts: List[Dict],
                                   fitness_scores: List[float]) -> str:
        """Construct prompt for vocabulary expansion"""
        examples = []
        for i, (prompt, fitness) in enumerate(zip(best_prompts[:5], fitness_scores[:5]), 1):
            examples.append(f"Prompt {i} (fitness: {fitness:.3f}):")
            examples.append(f"  comp: {', '.join(prompt.get('composition', []))}")
            examples.append(f"  light: {', '.join(prompt.get('lighting', []))}")
            examples.append(f"  style: {', '.join(prompt.get('style', []))}")
            examples.append("")

        examples_text = "\n".join(examples)

        return f"""Analyze these high-fitness prompts and generate {self.expansion_size} NEW modifiers:

{examples_text}

Generate terms that are:
1. Semantically related to what's working
2. Exploring adjacent regions
3. Filling gaps in the vocabulary
4. Diverse across categories

Output ONLY valid JSON (no markdown):
{{
  "composition": ["new1", "new2", ...],
  "lighting": [...],
  "style": [...],
  "quality": [...],
  "negative": [...]
}}"""

    # ==================== PRUNING ====================

    def prune_vocabulary(self, current_generation: int) -> int:
        """
        Remove unused modifiers

        Args:
            current_generation: Current generation number

        Returns:
            Number of modifiers pruned
        """
        print(f"\n{'='*70}")
        print(f"VOCABULARY PRUNING (Generation {current_generation})")
        print(f"{'='*70}")

        pruned_count = 0

        for block_name in self.vocabularies.keys():
            to_remove = []

            for modifier in self.vocabularies[block_name]:
                added_gen = self.generation_added[block_name].get(modifier, 0)
                if current_generation - added_gen < self.prune_threshold:
                    continue

                if self.usage_counts[block_name][modifier] == 0:
                    to_remove.append(modifier)

            for modifier in to_remove:
                self.vocabularies[block_name].remove(modifier)
                if modifier in self.synonym_map:
                    del self.synonym_map[modifier]
                pruned_count += 1

        if pruned_count > 0:
            print(f"Pruned {pruned_count} unused modifiers")
        else:
            print("No modifiers pruned")

        self._print_vocabulary_stats("AFTER PRUNING")

        return pruned_count

    # ==================== TRACKING ====================

    def track_usage(self, genome):
        """Track modifier usage in a genome"""
        self.usage_counts["composition"].update(genome.composition)
        self.usage_counts["lighting"].update(genome.lighting)
        self.usage_counts["style"].update(genome.style)
        self.usage_counts["quality"].update(genome.quality)
        self.usage_counts["negative"].update(genome.negative)

    def get_synonyms(self, modifier: str) -> List[str]:
        """Get synonyms for a modifier"""
        return self.synonym_map.get(modifier, [])

    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "total_uses": sum(sum(counts.values()) for counts in self.usage_counts.values()),
            "most_used": {
                block: counts.most_common(10)
                for block, counts in self.usage_counts.items()
            },
            "never_used": {
                block: [m for m in self.vocabularies[block] if counts[m] == 0]
                for block, counts in self.usage_counts.items()
            }
        }

    # ==================== PERSISTENCE ====================

    def save_vocabulary(self, filepath: str):
        """Save vocabulary state to JSON"""
        data = {
            "vocabularies": self.vocabularies,
            "synonyms": self.synonym_map,
            "usage_counts": {
                block: dict(counts) for block, counts in self.usage_counts.items()
            },
            "generation_added": self.generation_added,
            "stats": {
                "total_modifiers": sum(len(v) for v in self.vocabularies.values()),
                "total_uses": sum(sum(c.values()) for c in self.usage_counts.values())
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Vocabulary saved: {filepath}")

    def load_vocabulary(self, filepath: str):
        """Load vocabulary from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.vocabularies = data["vocabularies"]
        self.synonym_map = data.get("synonyms", {})
        self.usage_counts = {
            block: Counter(counts) for block, counts in data.get("usage_counts", {}).items()
        }
        self.generation_added = data.get("generation_added", {})

        print(f"✓ Vocabulary loaded: {filepath}")
        self._print_vocabulary_stats("LOADED VOCABULARY")

    # ==================== HELPERS ====================

    def _image_to_part(self, image: Image.Image):
        """Convert PIL Image to Vertex AI Part"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

        return self.Part.from_data(image_bytes, mime_type="image/jpeg")

    def _parse_vocabulary_response(self, response_text: str) -> Dict:
        """Parse LLM response for initial vocabulary"""
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        if "vocabularies" not in data:
            raise ValueError("Missing 'vocabularies' key")
        if "synonyms" not in data:
            data["synonyms"] = {}

        return data

    def _parse_expansion_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse LLM response for expansion"""
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        for block in ["composition", "lighting", "style", "quality", "negative"]:
            if block not in data:
                data[block] = []

        return data

    def _fallback_vocabulary(self) -> Dict[str, List[str]]:
        """Minimal static vocabulary as fallback"""
        from src.utils import create_block_vocabularies
        vocab = create_block_vocabularies()

        for block_name, modifiers in vocab.items():
            for modifier in modifiers:
                self.generation_added[block_name][modifier] = 0

        self.vocabularies = vocab
        return vocab

    def _print_vocabulary_stats(self, label: str):
        """Print vocabulary statistics"""
        print(f"\n{label}:")
        total = 0
        for block_name, modifiers in self.vocabularies.items():
            count = len(modifiers)
            total += count
            print(f"  {block_name:12s}: {count:4d} modifiers")
        print(f"  {'TOTAL':12s}: {total:4d} modifiers")
        print(f"  Synonym mappings: {len(self.synonym_map)}")
        print()
