"""
Block-structured genome for Experiment 2: Template-Based Generation
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import copy


@dataclass
class BlockGenome:
    """
    Block-structured genome for template-based generation

    Attributes:
        subject: User's content (NEVER mutated)
        composition: Composition block (evolvable)
        lighting: Lighting block (evolvable)
        style: Style block (evolvable)
        quality: Quality block (evolvable)
        negative: Negative prompt block (evolvable)
        fitness: Fitness score
    """
    subject: str
    composition: List[str] = field(default_factory=list)
    lighting: List[str] = field(default_factory=list)
    style: List[str] = field(default_factory=list)
    quality: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    fitness: float = 0.0

    def to_prompt(self) -> str:
        """
        Convert genome to full text prompt

        Returns:
            Complete prompt string
        """
        parts = [self.subject]

        # Add non-empty blocks
        for block in [self.composition, self.lighting, self.style, self.quality]:
            if block:
                parts.extend(block)

        return ", ".join(parts)

    def get_negative_prompt(self) -> str:
        """
        Get negative prompt string

        Returns:
            Negative prompt (or empty string if no negative modifiers)
        """
        return ", ".join(self.negative)

    def clone(self) -> 'BlockGenome':
        """
        Create a deep copy of this genome

        Returns:
            New BlockGenome instance with copied values
        """
        return BlockGenome(
            subject=self.subject,
            composition=copy.deepcopy(self.composition),
            lighting=copy.deepcopy(self.lighting),
            style=copy.deepcopy(self.style),
            quality=copy.deepcopy(self.quality),
            negative=copy.deepcopy(self.negative),
            fitness=self.fitness
        )

    def get_block(self, block_name: str) -> List[str]:
        """Get block by name"""
        return getattr(self, block_name)

    def set_block(self, block_name: str, values: List[str]):
        """Set block by name"""
        setattr(self, block_name, values)

    def __str__(self) -> str:
        """String representation"""
        return f"BlockGenome(fitness={self.fitness:.3f}, prompt='{self.to_prompt()[:80]}...')"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"BlockGenome(\n"
            f"  subject='{self.subject}',\n"
            f"  composition={self.composition},\n"
            f"  lighting={self.lighting},\n"
            f"  style={self.style},\n"
            f"  quality={self.quality},\n"
            f"  negative={self.negative},\n"
            f"  fitness={self.fitness:.3f}\n"
            f")"
        )


class BlockGenomeFactory:
    """Factory for creating BlockGenome instances"""

    def __init__(
        self,
        block_vocabularies: Dict[str, List[str]],
        max_per_block: int = 3
    ):
        """
        Initialize block genome factory

        Args:
            block_vocabularies: Dictionary mapping block names to vocabularies
            max_per_block: Maximum number of items per block
        """
        self.vocabularies = block_vocabularies
        self.max_per_block = max_per_block

        # Define evolvable blocks (subject is not included)
        self.evolvable_blocks = ['composition', 'lighting', 'style', 'quality', 'negative']

    def create_random(self, subject: str) -> BlockGenome:
        """
        Create a genome with random blocks

        Args:
            subject: User's subject/content (fixed)

        Returns:
            New BlockGenome with random blocks
        """
        genome = BlockGenome(subject=subject)

        for block_name in self.evolvable_blocks:
            vocab = self.vocabularies.get(block_name, [])
            if vocab:
                # Random number of items (0 to max)
                num_items = random.randint(0, min(self.max_per_block, len(vocab)))
                if num_items > 0:
                    items = random.sample(vocab, num_items)
                    genome.set_block(block_name, items)

        return genome

    def create_from_llm_seed(
        self,
        subject: str,
        llm_seed: Dict[str, List[str]]
    ) -> BlockGenome:
        """
        Create a genome from LLM-generated seed

        Args:
            subject: User's subject/content (fixed)
            llm_seed: Dictionary with seed values for each block

        Returns:
            New BlockGenome initialized from LLM seed
        """
        genome = BlockGenome(subject=subject)

        for block_name in self.evolvable_blocks:
            seed_values = llm_seed.get(block_name, [])

            # Truncate if exceeds max
            if len(seed_values) > self.max_per_block:
                seed_values = seed_values[:self.max_per_block]

            # Remove duplicates
            seed_values = list(dict.fromkeys(seed_values))

            genome.set_block(block_name, seed_values)

        return genome

    def create_hybrid(
        self,
        subject: str,
        llm_seed: Dict[str, List[str]]
    ) -> BlockGenome:
        """
        Create a genome with LLM seed + random fill

        Args:
            subject: User's subject/content (fixed)
            llm_seed: Dictionary with seed values for each block

        Returns:
            New BlockGenome with seeded + random values
        """
        genome = BlockGenome(subject=subject)

        for block_name in self.evolvable_blocks:
            seed_values = llm_seed.get(block_name, [])
            vocab = self.vocabularies.get(block_name, [])

            if not vocab:
                continue

            # Remove duplicates from seed
            seed_values = list(dict.fromkeys(seed_values))

            # Truncate seed if needed
            seed_values = seed_values[:self.max_per_block]

            # Fill remaining slots with random values
            block_values = seed_values.copy()
            remaining_slots = self.max_per_block - len(block_values)

            if remaining_slots > 0:
                # Get available values not in seed
                available = [v for v in vocab if v not in block_values]
                if available:
                    # Random number of additional items
                    num_to_add = random.randint(0, min(remaining_slots, len(available)))
                    if num_to_add > 0:
                        additional = random.sample(available, num_to_add)
                        block_values.extend(additional)

            genome.set_block(block_name, block_values)

        return genome

    def create_empty(self, subject: str) -> BlockGenome:
        """
        Create a genome with empty blocks (baseline)

        Args:
            subject: User's subject/content

        Returns:
            BlockGenome with empty blocks
        """
        return BlockGenome(subject=subject)
