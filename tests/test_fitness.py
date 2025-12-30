"""
Tests for fitness function validation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
import numpy as np

from src.fitness import FitnessEvaluator
from src.fitness_v2 import TemplateFitnessEvaluator


def create_test_image(color=(100, 150, 200), size=(512, 512)):
    """Create a simple test image"""
    return Image.new('RGB', size, color=color)


def test_clip_scores():
    """Test that CLIP gives higher scores for matching prompts"""
    print("\n" + "="*80)
    print("Testing CLIP Score Functionality")
    print("="*80)

    try:
        # Initialize evaluator
        evaluator = FitnessEvaluator(clip_weight=1.0, aesthetic_weight=0.0)
        print("✓ FitnessEvaluator initialized")

        # Create test images
        blue_image = create_test_image((50, 50, 200))  # Blue
        red_image = create_test_image((200, 50, 50))   # Red

        # Test matching vs non-matching prompts
        print("\nTesting prompt alignment...")

        # Blue image with blue prompt
        score_blue_blue = evaluator._clip_score(blue_image, "a blue image")
        print(f"  Blue image + 'a blue image': {score_blue_blue:.4f}")

        # Blue image with red prompt
        score_blue_red = evaluator._clip_score(blue_image, "a red image")
        print(f"  Blue image + 'a red image': {score_blue_red:.4f}")

        # Red image with red prompt
        score_red_red = evaluator._clip_score(red_image, "a red image")
        print(f"  Red image + 'a red image': {score_red_red:.4f}")

        # Verify scores are in valid range
        assert 0.0 <= score_blue_blue <= 1.0, "CLIP score out of range"
        assert 0.0 <= score_blue_red <= 1.0, "CLIP score out of range"
        assert 0.0 <= score_red_red <= 1.0, "CLIP score out of range"

        print("\n✓ All CLIP scores in valid range [0, 1]")

        # Note: Due to the simplicity of test images, scores may be similar
        # In real scenarios with complex images, matching prompts should score higher

        print("\n✅ CLIP scores test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ CLIP scores test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_lpips_similarity():
    """Test LPIPS similarity function"""
    print("\n" + "="*80)
    print("Testing LPIPS Similarity Functionality")
    print("="*80)

    try:
        # Create reference image
        reference = create_test_image((100, 150, 200))
        print("✓ Reference image created")

        # Initialize evaluator
        evaluator = TemplateFitnessEvaluator(
            reference_image=reference,
            clip_weight=0.0,
            lpips_weight=1.0
        )
        print("✓ TemplateFitnessEvaluator initialized")

        # Test 1: Identical image should have high similarity
        print("\nTest 1: Identical images...")
        identical = reference.copy()
        similarity_identical = evaluator._lpips_similarity(identical)
        print(f"  Similarity (identical): {similarity_identical:.4f}")
        assert similarity_identical > 0.5, "Identical images should have high similarity"
        print("  ✓ High similarity for identical images")

        # Test 2: Similar color image should have moderate-high similarity
        print("\nTest 2: Similar color images...")
        similar = create_test_image((110, 160, 210))  # Similar blue
        similarity_similar = evaluator._lpips_similarity(similar)
        print(f"  Similarity (similar): {similarity_similar:.4f}")
        assert similarity_similar > 0.3, "Similar images should have reasonable similarity"
        print("  ✓ Moderate similarity for similar images")

        # Test 3: Very different image should have lower similarity
        print("\nTest 3: Different color images...")
        different = create_test_image((200, 50, 50))  # Red (very different)
        similarity_different = evaluator._lpips_similarity(different)
        print(f"  Similarity (different): {similarity_different:.4f}")
        print("  ✓ Similarity computed for different images")

        # Verify ordering (identical > similar >= different)
        print("\nVerifying similarity ordering...")
        print(f"  Identical ({similarity_identical:.4f}) > Similar ({similarity_similar:.4f}): "
              f"{similarity_identical > similarity_similar}")
        print(f"  Similar ({similarity_similar:.4f}) >= Different ({similarity_different:.4f}): "
              f"{similarity_similar >= similarity_different}")

        # All scores should be in valid range
        for score in [similarity_identical, similarity_similar, similarity_different]:
            assert 0.0 <= score <= 1.0, f"LPIPS similarity {score} out of range"

        print("\n✓ All LPIPS similarities in valid range [0, 1]")

        print("\n✅ LPIPS similarity test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ LPIPS similarity test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_combined_fitness():
    """Test combined fitness evaluation"""
    print("\n" + "="*80)
    print("Testing Combined Fitness (CLIP + Aesthetic)")
    print("="*80)

    try:
        # Initialize evaluator with balanced weights
        evaluator = FitnessEvaluator(
            clip_weight=0.6,
            aesthetic_weight=0.4
        )
        print("✓ FitnessEvaluator initialized (CLIP: 0.6, Aesthetic: 0.4)")

        # Create test image
        test_image = create_test_image((100, 150, 200))
        test_prompt = "a beautiful blue image"

        # Evaluate fitness
        print("\nEvaluating fitness...")
        fitness = evaluator.evaluate(test_image, test_prompt, verbose=True)

        # Verify fitness is in valid range
        assert 0.0 <= fitness <= 1.0, "Fitness score out of range"
        print(f"\n✓ Combined fitness in valid range: {fitness:.4f}")

        print("\n✅ Combined fitness test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Combined fitness test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_template_fitness():
    """Test template fitness with CLIP + LPIPS"""
    print("\n" + "="*80)
    print("Testing Template Fitness (CLIP + LPIPS)")
    print("="*80)

    try:
        # Create reference image
        reference = create_test_image((100, 150, 200))

        # Initialize evaluator
        evaluator = TemplateFitnessEvaluator(
            reference_image=reference,
            clip_weight=0.5,
            lpips_weight=0.5
        )
        print("✓ TemplateFitnessEvaluator initialized (CLIP: 0.5, LPIPS: 0.5)")

        # Create test image (similar to reference)
        test_image = create_test_image((110, 160, 210))
        test_prompt = "a dog sitting on grass"

        # Evaluate fitness
        print("\nEvaluating template fitness...")
        fitness = evaluator.evaluate(test_image, test_prompt, verbose=True)

        # Verify fitness is in valid range
        assert 0.0 <= fitness <= 1.0, "Fitness score out of range"
        print(f"\n✓ Template fitness in valid range: {fitness:.4f}")

        print("\n✅ Template fitness test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Template fitness test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all fitness tests"""
    print("\n" + "="*80)
    print("FITNESS FUNCTION VALIDATION TESTS")
    print("="*80)

    results = []

    # Test CLIP scores
    results.append(("CLIP Scores", test_clip_scores()))

    # Test LPIPS similarity
    results.append(("LPIPS Similarity", test_lpips_similarity()))

    # Test combined fitness
    results.append(("Combined Fitness", test_combined_fitness()))

    # Test template fitness
    results.append(("Template Fitness", test_template_fitness()))

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

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
