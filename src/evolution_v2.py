"""
Block-aware genetic operators for Experiment 2
"""

import random
from typing import List, Optional, Dict
from src.genome_v2 import BlockGenome, BlockGenomeFactory


class BlockGeneticOperators:
    """Genetic operators for evolving BlockGenomes"""

    def __init__(
        self,
        factory: BlockGenomeFactory,
        vocabulary_manager=None,
        mutation_rate: float = 0.4,
        add_probability: float = 0.3,
        remove_probability: float = 0.2
    ):
        """
        Initialize block genetic operators

        Args:
            factory: BlockGenomeFactory for creating/modifying genomes
            vocabulary_manager: VocabularyManager for synonym-aware mutation (optional)
            mutation_rate: Probability of mutating each block item
            add_probability: Probability of adding a new item during mutation
            remove_probability: Probability of removing an item during mutation
        """
        self.factory = factory
        self.vocab_manager = vocabulary_manager
        self.mutation_rate = mutation_rate
        self.add_probability = add_probability
        self.remove_probability = remove_probability

    def tournament_selection(
        self,
        population: List[BlockGenome],
        k: int = 3
    ) -> BlockGenome:
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
        population: List[BlockGenome]
    ) -> BlockGenome:
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

    def mutate(self, genome: BlockGenome) -> BlockGenome:
        """
        Mutate a genome by modifying each block independently

        Subject block is NEVER mutated

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome (new instance)
        """
        mutated = genome.clone()

        # Mutate each evolvable block
        for block_name in self.factory.evolvable_blocks:
            current_block = mutated.get_block(block_name)
            vocab = self.factory.vocabularies.get(block_name, [])

            if vocab:
                new_block = self._mutate_block(
                    current_block,
                    vocab,
                    self.factory.max_per_block
                )
                mutated.set_block(block_name, new_block)

        return mutated

    def _mutate_block(
        self,
        block: List[str],
        vocab: List[str],
        max_items: int
    ) -> List[str]:
        """
        Mutate a single block

        Args:
            block: Current block values
            vocab: Vocabulary for this block
            max_items: Maximum items per block

        Returns:
            Mutated block
        """
        new_block = block.copy()

        # Mutate existing items
        for i in range(len(new_block)):
            if random.random() < self.mutation_rate:
                # Swap with a random item from vocab
                available = [v for v in vocab if v not in new_block]
                if available:
                    new_block[i] = random.choice(available)

        # Potentially add new items
        if len(new_block) < max_items and random.random() < self.add_probability:
            available = [v for v in vocab if v not in new_block]
            if available:
                num_to_add = random.randint(1, max_items - len(new_block))
                num_to_add = min(num_to_add, len(available))
                new_block.extend(random.sample(available, num_to_add))

        # Potentially remove items
        if len(new_block) > 0 and random.random() < self.remove_probability:
            num_to_remove = random.randint(1, len(new_block))
            for _ in range(num_to_remove):
                if new_block:
                    new_block.pop(random.randrange(len(new_block)))

        # Ensure no duplicates
        new_block = list(dict.fromkeys(new_block))

        return new_block

    def crossover(
        self,
        parent1: BlockGenome,
        parent2: BlockGenome
    ) -> BlockGenome:
        """
        Create offspring by swapping entire blocks between parents

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Offspring genome
        """
        offspring = parent1.clone()

        # For each block, randomly choose from parent1 or parent2
        for block_name in self.factory.evolvable_blocks:
            if random.random() < 0.5:
                # Take from parent2
                block = parent2.get_block(block_name)
                offspring.set_block(block_name, block.copy())

        return offspring

    def elitism(
        self,
        population: List[BlockGenome],
        elite_size: int
    ) -> List[BlockGenome]:
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

    # ==================== ADAPTIVE VOCABULARY METHODS ====================

    def mutate_with_synonyms(self, genome: BlockGenome) -> BlockGenome:
        """
        Mutate using synonym mappings when available (ADAPTIVE VOCABULARY)

        Strategy:
        1. For each modifier, with mutation_rate probability:
           a. Try synonyms first (50% chance if available) - semantic mutation
           b. Otherwise, random from vocabulary - explorative mutation
        2. Add/remove operations as before

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome
        """
        if self.vocab_manager is None:
            # Fall back to regular mutation
            return self.mutate(genome)

        mutated = genome.clone()

        # Mutate each block with synonym awareness
        mutated.composition = self._mutate_block_with_synonyms(
            mutated.composition, "composition"
        )
        mutated.lighting = self._mutate_block_with_synonyms(
            mutated.lighting, "lighting"
        )
        mutated.style = self._mutate_block_with_synonyms(
            mutated.style, "style"
        )
        mutated.quality = self._mutate_block_with_synonyms(
            mutated.quality, "quality"
        )
        mutated.negative = self._mutate_block_with_synonyms(
            mutated.negative, "negative"
        )

        return mutated

    def _mutate_block_with_synonyms(
        self,
        block: List[str],
        block_name: str
    ) -> List[str]:
        """
        Mutate a block using synonyms when available

        Args:
            block: Current block values
            block_name: Name of the block

        Returns:
            Mutated block
        """
        vocab = self.vocab_manager.vocabularies.get(block_name, [])
        if not vocab:
            return block

        new_block = []

        # Mutate existing modifiers
        for modifier in block:
            if random.random() < self.mutation_rate:
                # Try synonym-based mutation first (50% chance)
                synonyms = self.vocab_manager.get_synonyms(modifier)

                if synonyms and random.random() < 0.5:
                    # Use synonym (semantic mutation)
                    replacement = random.choice(synonyms)
                    if replacement not in new_block:
                        new_block.append(replacement)
                    else:
                        # Synonym already used, keep original
                        if modifier not in new_block:
                            new_block.append(modifier)
                else:
                    # Random from vocabulary (explorative mutation)
                    available = [m for m in vocab if m not in new_block and m != modifier]
                    if available:
                        new_block.append(random.choice(available))
                    else:
                        if modifier not in new_block:
                            new_block.append(modifier)
            else:
                # No mutation, keep original (avoid duplicates)
                if modifier not in new_block:
                    new_block.append(modifier)

        # Add new modifier (30% chance)
        max_per_block = self.factory.max_per_block
        if len(new_block) < max_per_block and random.random() < 0.3:
            available = [m for m in vocab if m not in new_block]
            if available:
                new_block.append(random.choice(available))

        # Remove modifier (20% chance, keep at least 1)
        if len(new_block) > 1 and random.random() < 0.2:
            new_block.pop(random.randint(0, len(new_block) - 1))

        return new_block


class BlockEvolutionEngine:
    """Evolution engine for block-based genomes"""

    def __init__(
        self,
        factory: BlockGenomeFactory,
        operators: BlockGeneticOperators,
        population_size: int = 10,
        elite_size: int = 1,
        selection_method: str = "tournament",
        use_crossover: bool = False
    ):
        """
        Initialize block evolution engine

        Args:
            factory: BlockGenomeFactory
            operators: BlockGeneticOperators
            population_size: Size of population
            elite_size: Number of elite genomes to preserve
            selection_method: 'tournament' or 'weighted'
            use_crossover: Whether to use crossover in addition to mutation
        """
        self.factory = factory
        self.operators = operators
        self.population_size = population_size
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.use_crossover = use_crossover

    def initialize_population(
        self,
        subject: str,
        llm_seeds: Optional[List[Dict[str, List[str]]]] = None
    ) -> List[BlockGenome]:
        """
        Initialize population with LLM seeds and random genomes

        Args:
            subject: User's subject/content
            llm_seeds: List of LLM-generated seed dictionaries

        Returns:
            Initial population
        """
        population = []

        # Create genomes from LLM seeds
        if llm_seeds:
            for seed in llm_seeds[:self.population_size]:
                genome = self.factory.create_from_llm_seed(subject, seed)
                population.append(genome)

        # Fill rest with random genomes
        while len(population) < self.population_size:
            genome = self.factory.create_random(subject)
            population.append(genome)

        return population

    def evolve_generation(
        self,
        population: List[BlockGenome]
    ) -> List[BlockGenome]:
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
            if self.use_crossover and random.random() < 0.3:
                # Crossover + mutation
                parent1 = self._select_parent(population)
                parent2 = self._select_parent(population)
                offspring = self.operators.crossover(parent1, parent2)
                offspring = self.operators.mutate(offspring)
            else:
                # Mutation only
                parent = self._select_parent(population)
                offspring = self.operators.mutate(parent)

            next_generation.append(offspring)

        return next_generation

    def _select_parent(self, population: List[BlockGenome]) -> BlockGenome:
        """Select parent using configured selection method"""
        if self.selection_method == "tournament":
            return self.operators.tournament_selection(population)
        else:
            return self.operators.weighted_random_selection(population)

    def get_best(self, population: List[BlockGenome]) -> BlockGenome:
        """
        Get the genome with highest fitness

        Args:
            population: Population of genomes

        Returns:
            Best genome
        """
        return max(population, key=lambda g: g.fitness)

    def get_diversity(self, population: List[BlockGenome]) -> Dict[str, float]:
        """
        Calculate diversity of population for each block

        Args:
            population: Population of genomes

        Returns:
            Dictionary mapping block name to diversity score
        """
        diversity = {}

        for block_name in self.factory.evolvable_blocks:
            unique_items = set()
            for genome in population:
                block = genome.get_block(block_name)
                unique_items.update(block)

            vocab_size = len(self.factory.vocabularies.get(block_name, []))
            if vocab_size > 0:
                diversity[block_name] = len(unique_items) / vocab_size
            else:
                diversity[block_name] = 0.0

        return diversity
