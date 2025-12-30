"""
Genetic operators and evolution engine for Experiment 1
"""

import random
from typing import List, Optional
from src.genome import PromptGenome, GenomeFactory


class GeneticOperators:
    """Genetic operators for evolving PromptGenomes"""

    def __init__(
        self,
        factory: GenomeFactory,
        vocabulary_manager=None,
        mutation_rate: float = 0.4,
        add_probability: float = 0.3,
        remove_probability: float = 0.2
    ):
        """
        Initialize genetic operators

        Args:
            factory: GenomeFactory for creating/modifying genomes
            vocabulary_manager: VocabularyManager for synonym-aware mutation (optional)
            mutation_rate: Probability of mutating each modifier
            add_probability: Probability of adding a new modifier during mutation
            remove_probability: Probability of removing a modifier during mutation
        """
        self.factory = factory
        self.vocab_manager = vocabulary_manager
        self.mutation_rate = mutation_rate
        self.add_probability = add_probability
        self.remove_probability = remove_probability

    def tournament_selection(
        self,
        population: List[PromptGenome],
        k: int = 3
    ) -> PromptGenome:
        """
        Select a genome using tournament selection

        Args:
            population: List of genomes
            k: Tournament size

        Returns:
            Selected genome
        """
        tournament = random.sample(population, min(k, len(population)))
        winner = max(tournament, key=lambda g: g.fitness)
        return winner

    def weighted_random_selection(
        self,
        population: List[PromptGenome]
    ) -> PromptGenome:
        """
        Select a genome using fitness-proportional selection

        Args:
            population: List of genomes

        Returns:
            Selected genome
        """
        # Handle negative or zero fitness
        min_fitness = min(g.fitness for g in population)
        if min_fitness <= 0:
            # Shift all fitness values to be positive
            weights = [g.fitness - min_fitness + 0.01 for g in population]
        else:
            weights = [g.fitness for g in population]

        total_fitness = sum(weights)
        if total_fitness == 0:
            # All fitnesses are equal, random selection
            return random.choice(population)

        # Normalize weights
        weights = [w / total_fitness for w in weights]

        # Random selection weighted by fitness
        selected = random.choices(population, weights=weights, k=1)[0]
        return selected

    def mutate(self, genome: PromptGenome) -> PromptGenome:
        """
        Mutate a genome by swapping, adding, or removing modifiers

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome (new instance)
        """
        # Use synonym-aware mutation if VocabularyManager is available
        if self.vocab_manager:
            return self.mutate_with_synonyms(genome)
        
        mutated = genome.clone()

        # Mutate positive modifiers
        mutated.positive_modifiers = self._mutate_modifiers(
            mutated.positive_modifiers,
            self.factory.modifier_vocab,
            self.factory.max_positive_modifiers
        )

        # Mutate negative modifiers
        mutated.negative_modifiers = self._mutate_modifiers(
            mutated.negative_modifiers,
            self.factory.negative_vocab,
            self.factory.max_negative_modifiers
        )

        return mutated
    
    def mutate_with_synonyms(self, genome: PromptGenome) -> PromptGenome:
        """
        Mutate using synonym mappings when available (ADAPTIVE VOCABULARY)
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome (new instance)
        """
        mutated = genome.clone()
        
        # Mutate positive modifiers with synonym awareness
        mutated.positive_modifiers = self._mutate_modifiers_with_synonyms(
            mutated.positive_modifiers,
            self.factory.modifier_vocab,
            self.factory.max_positive_modifiers
        )
        
        # Mutate negative modifiers with synonym awareness
        mutated.negative_modifiers = self._mutate_modifiers_with_synonyms(
            mutated.negative_modifiers,
            self.factory.negative_vocab,
            self.factory.max_negative_modifiers
        )
        
        return mutated
    
    def _mutate_modifiers_with_synonyms(
        self,
        modifiers: List[str],
        vocab: List[str],
        max_modifiers: int
    ) -> List[str]:
        """
        Mutate modifiers using synonyms when available
        
        Args:
            modifiers: Current modifiers
            vocab: Vocabulary to sample from
            max_modifiers: Maximum number of modifiers
            
        Returns:
            Mutated modifiers list
        """
        new_modifiers = []
        
        # Mutate existing modifiers
        for modifier in modifiers:
            if random.random() < self.mutation_rate:
                # Try synonym-based mutation first (50% chance if available)
                synonyms = self.vocab_manager.get_synonyms(modifier) if self.vocab_manager else []
                
                if synonyms and random.random() < 0.5:
                    # Use synonym (semantic mutation)
                    replacement = random.choice(synonyms)
                    if replacement not in new_modifiers and replacement in vocab:
                        new_modifiers.append(replacement)
                    else:
                        # Synonym not available, keep original
                        if modifier not in new_modifiers:
                            new_modifiers.append(modifier)
                else:
                    # Random from vocabulary (explorative mutation)
                    available = [m for m in vocab if m not in new_modifiers and m != modifier]
                    if available:
                        new_modifiers.append(random.choice(available))
                    else:
                        if modifier not in new_modifiers:
                            new_modifiers.append(modifier)
            else:
                # No mutation, keep original
                if modifier not in new_modifiers:
                    new_modifiers.append(modifier)
        
        # Potentially add new modifiers
        if len(new_modifiers) < max_modifiers and random.random() < self.add_probability:
            available = [m for m in vocab if m not in new_modifiers]
            if available:
                num_to_add = random.randint(1, max_modifiers - len(new_modifiers))
                num_to_add = min(num_to_add, len(available))
                new_modifiers.extend(random.sample(available, num_to_add))
        
        # Potentially remove modifiers
        if len(new_modifiers) > 0 and random.random() < self.remove_probability:
            num_to_remove = random.randint(1, len(new_modifiers))
            for _ in range(num_to_remove):
                if new_modifiers:
                    new_modifiers.pop(random.randrange(len(new_modifiers)))
        
        # Ensure no duplicates
        new_modifiers = list(dict.fromkeys(new_modifiers))
        
        return new_modifiers

    def _mutate_modifiers(
        self,
        modifiers: List[str],
        vocab: List[str],
        max_modifiers: int
    ) -> List[str]:
        """
        Mutate a list of modifiers

        Args:
            modifiers: Current modifiers
            vocab: Vocabulary to sample from
            max_modifiers: Maximum number of modifiers

        Returns:
            Mutated modifiers list
        """
        new_modifiers = modifiers.copy()

        # Iterate over each modifier and potentially mutate
        for i in range(len(new_modifiers)):
            if random.random() < self.mutation_rate:
                # Swap with a random modifier from vocab
                available = [m for m in vocab if m not in new_modifiers]
                if available:
                    new_modifiers[i] = random.choice(available)

        # Potentially add new modifiers
        if len(new_modifiers) < max_modifiers and random.random() < self.add_probability:
            available = [m for m in vocab if m not in new_modifiers]
            if available:
                num_to_add = random.randint(1, max_modifiers - len(new_modifiers))
                num_to_add = min(num_to_add, len(available))
                new_modifiers.extend(random.sample(available, num_to_add))

        # Potentially remove modifiers
        if len(new_modifiers) > 0 and random.random() < self.remove_probability:
            num_to_remove = random.randint(1, len(new_modifiers))
            for _ in range(num_to_remove):
                if new_modifiers:
                    new_modifiers.pop(random.randrange(len(new_modifiers)))

        # Ensure no duplicates
        new_modifiers = list(dict.fromkeys(new_modifiers))

        return new_modifiers

    def elitism(
        self,
        population: List[PromptGenome],
        elite_size: int
    ) -> List[PromptGenome]:
        """
        Select top genomes for elitism

        Args:
            population: Current population
            elite_size: Number of elite genomes to preserve

        Returns:
            List of elite genomes (cloned)
        """
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        elite = [g.clone() for g in sorted_pop[:elite_size]]
        return elite


class EvolutionEngine:
    """Evolution engine for running genetic algorithm"""

    def __init__(
        self,
        factory: GenomeFactory,
        operators: GeneticOperators,
        population_size: int = 10,
        elite_size: int = 1,
        selection_method: str = "tournament"
    ):
        """
        Initialize evolution engine

        Args:
            factory: GenomeFactory
            operators: GeneticOperators
            population_size: Size of population
            elite_size: Number of elite genomes to preserve
            selection_method: 'tournament' or 'weighted'
        """
        self.factory = factory
        self.operators = operators
        self.population_size = population_size
        self.elite_size = elite_size
        self.selection_method = selection_method

    def initialize_population(
        self,
        base_prompt: str,
        seed_modifiers: Optional[List[str]] = None
    ) -> List[PromptGenome]:
        """
        Initialize population with mix of seeded and random genomes

        Args:
            base_prompt: Base prompt for all genomes
            seed_modifiers: Optional seed modifiers for initial population

        Returns:
            Initial population
        """
        population = []

        # If seed modifiers provided, create 50% seeded genomes
        if seed_modifiers:
            num_seeded = self.population_size // 2
            for _ in range(num_seeded):
                genome = self.factory.create_seeded(base_prompt, seed_modifiers)
                population.append(genome)

        # Fill rest with random genomes
        while len(population) < self.population_size:
            genome = self.factory.create_random(base_prompt)
            population.append(genome)

        return population

    def evolve_generation(
        self,
        population: List[PromptGenome]
    ) -> List[PromptGenome]:
        """
        Create next generation from current population

        Args:
            population: Current population (with fitness scores)

        Returns:
            Next generation population
        """
        next_generation = []

        # Preserve elite genomes
        elite = self.operators.elitism(population, self.elite_size)
        next_generation.extend(elite)

        # Generate offspring to fill rest of population
        while len(next_generation) < self.population_size:
            # Select parent
            if self.selection_method == "tournament":
                parent = self.operators.tournament_selection(population)
            else:
                parent = self.operators.weighted_random_selection(population)

            # Mutate to create offspring
            offspring = self.operators.mutate(parent)

            next_generation.append(offspring)

        return next_generation

    def get_best(self, population: List[PromptGenome]) -> PromptGenome:
        """
        Get the genome with highest fitness

        Args:
            population: Population of genomes

        Returns:
            Best genome
        """
        return max(population, key=lambda g: g.fitness)

    def get_diversity(self, population: List[PromptGenome]) -> float:
        """
        Calculate diversity of population based on unique modifiers

        Args:
            population: Population of genomes

        Returns:
            Diversity score (0-1)
        """
        all_modifiers = set()
        for genome in population:
            all_modifiers.update(genome.positive_modifiers)
            all_modifiers.update(genome.negative_modifiers)

        # Diversity = number of unique modifiers / total vocab size
        total_vocab_size = len(self.factory.modifier_vocab) + len(self.factory.negative_vocab)
        diversity = len(all_modifiers) / total_vocab_size

        return diversity
