"""
Tests for model API connectivity
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import get_model, FluxSchnellModel, QwenImageModel
from PIL import Image


def test_flux_schnell():
    """Test Flux-Schnell model connectivity"""
    print("\n" + "="*80)
    print("Testing Flux-Schnell Model")
    print("="*80)

    try:
        model = get_model("flux-schnell")
        print(f"✓ Model initialized: {model.__class__.__name__}")

        # Generate test image
        print("Generating test image...")
        image, metadata = model.generate(
            "A cat sitting on a roof",
            seed=42,
            num_inference_steps=4
        )

        print(f"✓ Image generated: {image.size}")
        print(f"  Metadata: {metadata.get('timings', {})}")

        # Save test image
        output_dir = Path(__file__).parent.parent / "data" / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "test_flux_schnell.jpg"
        image.save(output_path)
        print(f"✓ Test image saved: {output_path}")

        # Verify image
        assert isinstance(image, Image.Image), "Output is not a PIL Image"
        assert image.size[0] > 0 and image.size[1] > 0, "Image has invalid dimensions"
        assert output_path.exists(), "Image file was not saved"

        print("\n✅ Flux-Schnell test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Flux-Schnell test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_image():
    """Test Qwen-Image model connectivity"""
    print("\n" + "="*80)
    print("Testing Qwen-Image Model")
    print("="*80)

    try:
        model = get_model("qwen-image")
        print(f"✓ Model initialized: {model.__class__.__name__}")

        # Generate test image with negative prompt
        print("Generating test image with negative prompt...")
        image, metadata = model.generate(
            "A beautiful sunset over mountains",
            negative_prompt="blurry, low quality",
            seed=42,
            num_inference_steps=20  # Reduced for testing
        )

        print(f"✓ Image generated: {image.size}")
        print(f"  Metadata: {metadata.get('timings', {})}")

        # Save test image
        output_dir = Path(__file__).parent.parent / "data" / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "test_qwen_image.jpg"
        image.save(output_path)
        print(f"✓ Test image saved: {output_path}")

        # Verify image
        assert isinstance(image, Image.Image), "Output is not a PIL Image"
        assert image.size[0] > 0 and image.size[1] > 0, "Image has invalid dimensions"
        assert output_path.exists(), "Image file was not saved"

        print("\n✅ Qwen-Image test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Qwen-Image test FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all model tests"""
    print("\n" + "="*80)
    print("MODEL API CONNECTIVITY TESTS")
    print("="*80)

    results = []

    # Test Flux-Schnell
    results.append(("Flux-Schnell", test_flux_schnell()))

    # Test Qwen-Image (optional, slower)
    # Uncomment to test:
    # results.append(("Qwen-Image", test_qwen_image()))

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
        print("\n⚠️  Some tests failed. Please check API keys and connectivity.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
