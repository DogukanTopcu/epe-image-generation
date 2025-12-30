"""
Tests for Gemini LLM integration
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from src.llm_prompt_generator import get_prompt_generator, GeminiPromptGenerator, DummyPromptGenerator


def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    # Create a gradient image for more interesting analysis
    import numpy as np
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        arr[i, :, :] = [100 + i // 4, 150, 200 - i // 4]
    return Image.fromarray(arr)


def test_dummy_generator():
    """Test DummyPromptGenerator (no API calls)"""
    print("\n" + "="*80)
    print("Testing DummyPromptGenerator")
    print("="*80)

    try:
        # Initialize dummy generator
        generator = DummyPromptGenerator()
        print("✓ DummyPromptGenerator initialized")

        # Create test image
        reference = create_test_image()

        # Analyze reference
        print("\nGenerating prompts...")
        prompts = generator.analyze_reference_image(
            reference=reference,
            user_subject="a golden retriever",
            num_variations=3
        )

        print(f"✓ Generated {len(prompts)} prompt variations")

        # Verify structure
        assert len(prompts) == 3, "Should generate requested number of variations"

        for i, prompt in enumerate(prompts):
            print(f"\nVariation {i+1}:")
            assert isinstance(prompt, dict), "Each prompt should be a dictionary"

            required_blocks = ['composition', 'lighting', 'style', 'quality', 'negative']
            for block in required_blocks:
                assert block in prompt, f"Missing block: {block}"
                assert isinstance(prompt[block], list), f"Block {block} should be a list"
                print(f"  {block}: {prompt[block]}")

        print("\n✓ All prompts have correct structure")

        # Test seed generation
        print("\nGenerating seed prompts for population...")
        seeds = generator.generate_seed_prompts(
            reference=reference,
            user_subject="a golden retriever",
            population_size=5
        )

        assert len(seeds) == 5, "Should generate requested population size"
        print(f"✓ Generated {len(seeds)} seed prompts")

        print("\n✅ DummyPromptGenerator test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ DummyPromptGenerator test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gemini_analysis():
    """Test GeminiPromptGenerator (requires Vertex AI setup)"""
    print("\n" + "="*80)
    print("Testing GeminiPromptGenerator")
    print("="*80)

    # Check if Vertex AI is configured
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set, skipping Gemini test")
        print("   Set up Vertex AI credentials to test Gemini integration")
        return None  # Skip test

    try:
        # Initialize Gemini generator
        generator = get_prompt_generator(use_llm=True)

        if isinstance(generator, DummyPromptGenerator):
            print("⚠️  Gemini initialization failed, using DummyPromptGenerator")
            return None  # Skip test

        print(f"✓ GeminiPromptGenerator initialized")

        # Create test image
        reference = create_test_image()

        # Analyze reference
        print("\nAnalyzing reference image with Gemini...")
        prompts = generator.analyze_reference_image(
            reference=reference,
            user_subject="a golden retriever",
            num_variations=3
        )

        print(f"✓ Generated {len(prompts)} prompt variations")

        # Verify structure
        assert len(prompts) >= 1, "Should generate at least one variation"

        for i, prompt in enumerate(prompts[:3]):  # Show first 3
            print(f"\nVariation {i+1}:")
            assert isinstance(prompt, dict), "Each prompt should be a dictionary"

            required_blocks = ['composition', 'lighting', 'style', 'quality', 'negative']
            for block in required_blocks:
                if block in prompt:
                    print(f"  {block}: {prompt[block]}")

        print("\n✓ Gemini successfully analyzed image and generated prompts")

        # Test seed generation
        print("\nGenerating seed prompts...")
        seeds = generator.generate_seed_prompts(
            reference=reference,
            user_subject="a golden retriever",
            population_size=5
        )

        assert len(seeds) == 5, "Should generate requested population size"
        print(f"✓ Generated {len(seeds)} seed prompts")

        print("\n✅ GeminiPromptGenerator test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ GeminiPromptGenerator test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_factory_function():
    """Test get_prompt_generator factory function"""
    print("\n" + "="*80)
    print("Testing Prompt Generator Factory")
    print("="*80)

    try:
        # Test with use_llm=False
        print("Testing factory with use_llm=False...")
        generator_dummy = get_prompt_generator(use_llm=False)
        assert isinstance(generator_dummy, DummyPromptGenerator)
        print("✓ Returns DummyPromptGenerator when use_llm=False")

        # Test with use_llm=True (may fallback to Dummy if Vertex AI not configured)
        print("\nTesting factory with use_llm=True...")
        generator_llm = get_prompt_generator(use_llm=True)
        print(f"✓ Returns {generator_llm.__class__.__name__}")

        print("\n✅ Factory function test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Factory function test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all LLM generator tests"""
    print("\n" + "="*80)
    print("LLM PROMPT GENERATOR TESTS")
    print("="*80)

    results = []

    # Test dummy generator (always available)
    results.append(("DummyPromptGenerator", test_dummy_generator()))

    # Test factory function
    results.append(("Factory Function", test_factory_function()))

    # Test Gemini (optional, requires setup)
    gemini_result = test_gemini_analysis()
    if gemini_result is not None:
        results.append(("GeminiPromptGenerator", gemini_result))
    else:
        print("\n⚠️  Gemini test skipped (no credentials configured)")

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

    print("\nNote: To test Gemini integration, set up Vertex AI credentials:")
    print("  1. Set GOOGLE_CLOUD_PROJECT environment variable")
    print("  2. Set GOOGLE_APPLICATION_CREDENTIALS to service account key path")
    print("  3. Ensure Vertex AI API is enabled in your project")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
