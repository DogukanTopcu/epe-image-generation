"""
Tests for adaptive vocabulary manager
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from src.vocabulary_manager import VocabularyManager
from src.genome_v2 import BlockGenomeFactory


def create_test_image(size=(512, 512)):
    """Create a test image"""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        arr[i, :, :] = [100 + i // 4, 150, 200 - i // 4]
    return Image.fromarray(arr)


def test_initialization_fallback():
    """Test fallback vocabulary initialization"""
    print("\n" + "="*80)
    print("Test 1: Fallback Initialization (No LLM)")
    print("="*80)

    try:
        vocab_manager = VocabularyManager(use_llm=False)
        print("✓ VocabularyManager initialized with LLM disabled")

        vocabularies = vocab_manager.initialize_vocabulary()

        # Verify structure
        assert 'composition' in vocabularies
        assert 'lighting' in vocabularies
        assert 'style' in vocabularies
        assert 'quality' in vocabularies
        assert 'negative' in vocabularies

        print(f"✓ All required blocks present")

        # Verify non-empty
        for block_name, modifiers in vocabularies.items():
            assert len(modifiers) > 0, f"Block {block_name} is empty"
            print(f"  {block_name}: {len(modifiers)} modifiers")

        print("\n✅ Fallback initialization test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Fallback initialization test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_initialization_with_llm():
    """Test LLM-based initialization (requires Vertex AI)"""
    print("\n" + "="*80)
    print("Test 2: LLM-Based Initialization")
    print("="*80)

    # Check if credentials available
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set, skipping LLM test")
        return None

    try:
        vocab_manager = VocabularyManager(
            use_llm=True,
            initial_size=100  # Smaller for testing
        )

        if vocab_manager.model is None:
            print("⚠️  LLM initialization failed, skipping")
            return None

        print("✓ VocabularyManager initialized with LLM")

        # Generate vocabulary
        test_image = create_test_image()
        vocabularies = vocab_manager.initialize_vocabulary(
            reference_image=test_image,
            domain_description="test photography"
        )

        # Verify structure
        total_size = sum(len(v) for v in vocabularies.values())
        print(f"✓ Generated {total_size} total modifiers")

        assert total_size > 0, "No modifiers generated"

        # Check synonym mappings
        print(f"✓ Synonym mappings: {len(vocab_manager.synonym_map)}")

        print("\n✅ LLM initialization test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ LLM initialization test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_usage_tracking():
    """Test usage tracking functionality"""
    print("\n" + "="*80)
    print("Test 3: Usage Tracking")
    print("="*80)

    try:
        vocab_manager = VocabularyManager(use_llm=False)
        vocab_manager.initialize_vocabulary()

        # Create a genome and track usage
        factory = BlockGenomeFactory(vocab_manager.vocabularies, max_per_block=3)
        genome = factory.create_random("test subject")

        # Track usage
        vocab_manager.track_usage(genome)

        # Verify tracking
        stats = vocab_manager.get_usage_stats()

        assert 'total_uses' in stats
        assert 'most_used' in stats
        assert 'never_used' in stats

        print(f"✓ Total uses tracked: {stats['total_uses']}")
        print(f"✓ Usage statistics generated")

        print("\n✅ Usage tracking test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Usage tracking test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_synonym_retrieval():
    """Test synonym retrieval"""
    print("\n" + "="*80)
    print("Test 4: Synonym Retrieval")
    print("="*80)

    try:
        vocab_manager = VocabularyManager(use_llm=False)
        vocab_manager.initialize_vocabulary()

        # Add some test synonyms
        vocab_manager.synonym_map['test_term'] = ['synonym1', 'synonym2', 'synonym3']

        # Retrieve synonyms
        synonyms = vocab_manager.get_synonyms('test_term')

        assert len(synonyms) == 3
        assert 'synonym1' in synonyms

        print(f"✓ Retrieved {len(synonyms)} synonyms for 'test_term'")

        # Test non-existent term
        empty = vocab_manager.get_synonyms('nonexistent')
        assert len(empty) == 0

        print(f"✓ Returns empty list for non-existent terms")

        print("\n✅ Synonym retrieval test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Synonym retrieval test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pruning():
    """Test vocabulary pruning"""
    print("\n" + "="*80)
    print("Test 5: Vocabulary Pruning")
    print("="*80)

    try:
        vocab_manager = VocabularyManager(use_llm=False, prune_threshold=5)
        vocab_manager.initialize_vocabulary()

        initial_size = sum(len(v) for v in vocab_manager.vocabularies.values())
        print(f"Initial vocabulary size: {initial_size}")

        # Mark all as added at generation 0
        for block_name in vocab_manager.vocabularies.keys():
            for modifier in vocab_manager.vocabularies[block_name]:
                vocab_manager.generation_added[block_name][modifier] = 0

        # Prune at generation 10 (should remove items not used for 5+ generations)
        pruned_count = vocab_manager.prune_vocabulary(current_generation=10)

        final_size = sum(len(v) for v in vocab_manager.vocabularies.values())

        print(f"Pruned {pruned_count} modifiers")
        print(f"Final vocabulary size: {final_size}")

        # Should have pruned something
        assert pruned_count > 0, "Pruning didn't remove any modifiers"
        assert final_size < initial_size, "Vocabulary size didn't decrease"

        print("✓ Pruning successfully removed unused modifiers")

        print("\n✅ Pruning test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Pruning test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_persistence():
    """Test save/load functionality"""
    print("\n" + "="*80)
    print("Test 6: Persistence (Save/Load)")
    print("="*80)

    try:
        vocab_manager = VocabularyManager(use_llm=False)
        vocab_manager.initialize_vocabulary()

        # Add usage data
        vocab_manager.usage_counts['composition']['wide angle'] = 5
        vocab_manager.usage_counts['lighting']['soft lighting'] = 3

        # Save
        save_path = '../data/results/test_vocab.json'
        vocab_manager.save_vocabulary(save_path)

        assert os.path.exists(save_path), "Vocabulary file not created"
        print(f"✓ Vocabulary saved to {save_path}")

        # Load into new manager
        new_manager = VocabularyManager(use_llm=False)
        new_manager.load_vocabulary(save_path)

        # Verify loaded correctly
        assert len(new_manager.vocabularies) == len(vocab_manager.vocabularies)

        for block_name in vocab_manager.vocabularies.keys():
            assert len(new_manager.vocabularies[block_name]) == len(vocab_manager.vocabularies[block_name])

        print("✓ Vocabulary loaded successfully")

        # Verify usage counts preserved
        assert new_manager.usage_counts['composition']['wide angle'] == 5
        assert new_manager.usage_counts['lighting']['soft lighting'] == 3

        print("✓ Usage counts preserved")

        # Cleanup
        os.remove(save_path)

        print("\n✅ Persistence test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Persistence test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all vocabulary manager tests"""
    print("\n" + "="*80)
    print("VOCABULARY MANAGER TESTS")
    print("="*80)

    results = []

    # Test 1: Fallback initialization
    results.append(("Fallback Initialization", test_initialization_fallback()))

    # Test 2: LLM initialization (optional)
    llm_result = test_initialization_with_llm()
    if llm_result is not None:
        results.append(("LLM Initialization", llm_result))
    else:
        print("\n⚠️  LLM initialization test skipped (no credentials)\n")

    # Test 3: Usage tracking
    results.append(("Usage Tracking", test_usage_tracking()))

    # Test 4: Synonym retrieval
    results.append(("Synonym Retrieval", test_synonym_retrieval()))

    # Test 5: Pruning
    results.append(("Pruning", test_pruning()))

    # Test 6: Persistence
    results.append(("Persistence", test_persistence()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

    print("\nNote: To test LLM features, configure Vertex AI credentials:")
    print("  - Set GOOGLE_CLOUD_PROJECT environment variable")
    print("  - Set GOOGLE_APPLICATION_CREDENTIALS to service account key")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
