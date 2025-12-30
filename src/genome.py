"""
Genome representation for Experiment 1: Prompt Enhancement
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
import copy


@dataclass
class PromptGenome:
    """
    Genome for prompt enhancement experiment

    Attributes:
        base_prompt: Fixed base prompt (never mutated)
        positive_modifiers: List of positive style modifiers (evolvable)
        negative_modifiers: List of negative modifiers (evolvable)
        fitness: Fitness score
    """
    base_prompt: str
    positive_modifiers: List[str] = field(default_factory=list)
    negative_modifiers: List[str] = field(default_factory=list)
    fitness: float = 0.0

    def to_prompt(self) -> str:
        """
        Convert genome to full text prompt

        Returns:
            Complete prompt string
        """
        if not self.positive_modifiers:
            return self.base_prompt

        modifiers_str = ", ".join(self.positive_modifiers)
        return f"{self.base_prompt}, {modifiers_str}"

    def get_negative_prompt(self) -> str:
        """
        Get negative prompt string

        Returns:
            Negative prompt (or empty string if no negative modifiers)
        """
        return ", ".join(self.negative_modifiers)

    def clone(self) -> 'PromptGenome':
        """
        Create a deep copy of this genome

        Returns:
            New PromptGenome instance with copied values
        """
        return PromptGenome(
            base_prompt=self.base_prompt,
            positive_modifiers=copy.deepcopy(self.positive_modifiers),
            negative_modifiers=copy.deepcopy(self.negative_modifiers),
            fitness=self.fitness
        )

    def __str__(self) -> str:
        """String representation"""
        return f"PromptGenome(fitness={self.fitness:.3f}, prompt='{self.to_prompt()}')"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"PromptGenome(\n"
            f"  base='{self.base_prompt}',\n"
            f"  pos_mods={self.positive_modifiers},\n"
            f"  neg_mods={self.negative_modifiers},\n"
            f"  fitness={self.fitness:.3f}\n"
            f")"
        )


class GenomeFactory:
    """Factory for creating PromptGenome instances"""

    def __init__(
        self,
        modifier_vocab: List[str],
        negative_vocab: List[str],
        max_positive_modifiers: int = 8,
        max_negative_modifiers: int = 4
    ):
        """
        Initialize genome factory

        Args:
            modifier_vocab: Vocabulary of positive modifiers
            negative_vocab: Vocabulary of negative modifiers
            max_positive_modifiers: Maximum number of positive modifiers
            max_negative_modifiers: Maximum number of negative modifiers
        """
        self.modifier_vocab = modifier_vocab
        self.negative_vocab = negative_vocab
        self.max_positive_modifiers = max_positive_modifiers
        self.max_negative_modifiers = max_negative_modifiers

    def create_random(self, base_prompt: str) -> PromptGenome:
        """
        Create a genome with random modifiers

        Args:
            base_prompt: Base prompt (fixed)

        Returns:
            New PromptGenome with random modifiers
        """
        # Random number of positive modifiers (1 to max)
        num_positive = random.randint(1, self.max_positive_modifiers)
        positive_modifiers = random.sample(self.modifier_vocab, num_positive)

        # Random number of negative modifiers (0 to max)
        num_negative = random.randint(0, self.max_negative_modifiers)
        negative_modifiers = random.sample(self.negative_vocab, num_negative)

        return PromptGenome(
            base_prompt=base_prompt,
            positive_modifiers=positive_modifiers,
            negative_modifiers=negative_modifiers
        )

    def create_seeded(
        self,
        base_prompt: str,
        seed_modifiers: Optional[List[str]] = None,
        seed_negative: Optional[List[str]] = None
    ) -> PromptGenome:
        """
        Create a genome with specific seed modifiers, filling remaining slots randomly

        Args:
            base_prompt: Base prompt (fixed)
            seed_modifiers: Initial positive modifiers to include
            seed_negative: Initial negative modifiers to include

        Returns:
            New PromptGenome with seeded + random modifiers
        """
        seed_modifiers = seed_modifiers or []
        seed_negative = seed_negative or []

        # Ensure no duplicates in seed
        seed_modifiers = list(dict.fromkeys(seed_modifiers))
        seed_negative = list(dict.fromkeys(seed_negative))

        # Truncate if exceeds max
        seed_modifiers = seed_modifiers[:self.max_positive_modifiers]
        seed_negative = seed_negative[:self.max_negative_modifiers]

        # Fill remaining slots with random modifiers
        positive_modifiers = seed_modifiers.copy()
        remaining_positive_slots = self.max_positive_modifiers - len(positive_modifiers)

        if remaining_positive_slots > 0:
            # Get available modifiers (not already in seed)
            available = [m for m in self.modifier_vocab if m not in positive_modifiers]
            # Randomly decide how many more to add (0 to remaining slots)
            num_to_add = random.randint(0, min(remaining_positive_slots, len(available)))
            if num_to_add > 0:
                additional = random.sample(available, num_to_add)
                positive_modifiers.extend(additional)

        # Same for negative modifiers
        negative_modifiers = seed_negative.copy()
        remaining_negative_slots = self.max_negative_modifiers - len(negative_modifiers)

        if remaining_negative_slots > 0:
            available = [m for m in self.negative_vocab if m not in negative_modifiers]
            num_to_add = random.randint(0, min(remaining_negative_slots, len(available)))
            if num_to_add > 0:
                additional = random.sample(available, num_to_add)
                negative_modifiers.extend(additional)

        return PromptGenome(
            base_prompt=base_prompt,
            positive_modifiers=positive_modifiers,
            negative_modifiers=negative_modifiers
        )

    def create_empty(self, base_prompt: str) -> PromptGenome:
        """
        Create a genome with no modifiers (baseline)

        Args:
            base_prompt: Base prompt

        Returns:
            PromptGenome with no modifiers
        """
        return PromptGenome(
            base_prompt=base_prompt,
            positive_modifiers=[],
            negative_modifiers=[]
        )
