"""Test template analyzer."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from PIL import Image
from src.template_analyzer import TemplateAnalyzer
import json

load_dotenv()


def test_analysis():
    """Test TemplateAnalyzer functionality."""
    print("\n" + "="*80)
    print("TESTING TEMPLATE ANALYZER")
    print("="*80)

    # Initialize analyzer
    analyzer = TemplateAnalyzer(use_llm=True)

    # Load reference image
    reference_path = "data/reference_templates/template_1.png"
    if not os.path.exists(reference_path):
        # Try jpg
        reference_path = "data/reference_templates/template_1.jpg"

    if not os.path.exists(reference_path):
        print(f"ERROR: Reference image not found at {reference_path}")
        print("Please ensure a reference template exists")
        return False

    print(f"\nLoading reference: {reference_path}")
    reference = Image.open(reference_path)
    print(f"Image size: {reference.size}")

    # Test 1: Template Analysis
    print("\n" + "="*80)
    print("TEST 1: Template Analysis")
    print("="*80)

    try:
        analysis = analyzer.analyze_template(reference)

        # Verify required fields
        assert "lighting" in analysis, "Missing 'lighting' in analysis"
        assert "composition" in analysis, "Missing 'composition' in analysis"
        assert "aspect_ratio" in analysis, "Missing 'aspect_ratio' in analysis"
        assert "critical_elements" in analysis, "Missing 'critical_elements' in analysis"

        print("\n✅ TEST 1 PASSED: Analysis completed successfully")
        print(f"   Aspect ratio: {analysis['aspect_ratio']}")
        print(f"   Lighting type: {analysis['lighting'].get('type', 'N/A')}")
        print(f"   Critical elements: {len(analysis.get('critical_elements', []))}")

        # Save analysis for inspection
        os.makedirs("../data/results", exist_ok=True)
        with open("../data/results/test_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"   Saved analysis to: ../data/results/test_analysis.json")

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Template-Matching Prompt Generation
    print("\n" + "="*80)
    print("TEST 2: Template-Matching Prompt Generation")
    print("="*80)

    try:
        test_subjects = ["a young woman", "a cat", "an elderly man"]

        for subject in test_subjects:
            prompt = analyzer.generate_template_matching_prompt(analysis, subject)

            # Verify prompt quality
            assert len(prompt) > 20, f"Prompt too short: {prompt}"
            assert subject in prompt.lower(), f"Subject '{subject}' not in prompt"

            print(f"\n✅ Generated for '{subject}':")
            print(f"   {prompt}")

        print("\n✅ TEST 2 PASSED: Prompt generation successful")

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Specialized Vocabulary Generation
    print("\n" + "="*80)
    print("TEST 3: Specialized Vocabulary Generation")
    print("="*80)

    try:
        vocab = analyzer.generate_specialized_vocabulary(analysis, size=200)

        # Verify structure
        required_blocks = ["lighting", "composition", "style", "quality", "negative"]
        for block in required_blocks:
            assert block in vocab, f"Missing block: {block}"
            assert len(vocab[block]) > 0, f"Block '{block}' is empty"

        total = sum(len(v) for v in vocab.values())

        print(f"\n✅ TEST 3 PASSED: Vocabulary generation successful")
        print(f"   Total modifiers: {total}")
        print(f"   Lighting: {len(vocab['lighting'])} terms")
        print(f"   Composition: {len(vocab['composition'])} terms")
        print(f"   Style: {len(vocab['style'])} terms")
        print(f"   Quality: {len(vocab['quality'])} terms")
        print(f"   Negative: {len(vocab['negative'])} terms")

        # Save vocabulary for inspection
        with open("../data/results/test_vocabulary.json", 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"   Saved vocabulary to: ../data/results/test_vocabulary.json")

        # Print sample terms
        print(f"\n   Sample lighting terms: {vocab['lighting'][:5]}")
        print(f"   Sample composition terms: {vocab['composition'][:5]}")

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)
    print("\n✓ Template analysis working")
    print("✓ Prompt generation working")
    print("✓ Vocabulary generation working")
    print("\nYou can now integrate TemplateAnalyzer into the notebook!")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = test_analysis()
    sys.exit(0 if success else 1)
