"""
Advanced template analysis for structure-aware prompt generation.
Analyzes reference image deeply and generates template-matching prompts.
"""
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
from io import BytesIO
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

load_dotenv()


class TemplateAnalyzer:
    """
    Deep analysis of reference templates for structure-preserving generation.

    Key capabilities:
    - Analyze reference image lighting, composition, style in detail
    - Generate template-matching prompts for user subjects
    - Create specialized vocabularies based on template characteristics
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize template analyzer.

        Args:
            use_llm: If True, use Gemini for analysis. If False, use fallback.
        """
        self.use_llm = use_llm

        if use_llm:
            try:
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

                if not project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECT not set")

                vertexai.init(project=project_id, location=location)
                self.model = GenerativeModel("gemini-2.0-flash-exp")
                print(f"TemplateAnalyzer initialized with Gemini Vision API (project: {project_id})")
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
                print("Falling back to basic analysis")
                self.model = None
                self.use_llm = False
        else:
            self.model = None
            print("TemplateAnalyzer: LLM disabled")

    def analyze_template(self, reference_image: Image.Image) -> Dict:
        """
        Perform comprehensive analysis of reference template.

        Analyzes:
        - Lighting (direction, quality, colors, special techniques)
        - Composition (framing, angle, focal length, depth)
        - Background (type, color, treatment)
        - Style (photography type, mood, aesthetic)
        - Color palette
        - Technical specifications

        Args:
            reference_image: PIL Image of reference template

        Returns:
            Analysis dictionary with all visual characteristics
        """
        print("\n" + "="*70)
        print("DEEP TEMPLATE ANALYSIS")
        print("="*70)

        if not self.use_llm or self.model is None:
            print("Using fallback analysis (LLM unavailable)")
            return self._fallback_analysis(reference_image)

        try:
            # Detect aspect ratio
            aspect_ratio = self._detect_aspect_ratio(reference_image)
            print(f"Detected aspect ratio: {aspect_ratio}")

            # Construct detailed analysis prompt
            prompt_text = self._construct_analysis_prompt()

            # Convert image to Vertex AI format
            image_part = self._image_to_part(reference_image)

            # Call Gemini Vision
            print("Analyzing reference with Gemini Vision API...")
            print("This may take 10-20 seconds...")

            response = self.model.generate_content(
                [image_part, prompt_text],
                generation_config={"temperature": 0.1}  # Low temperature for consistency
            )

            # Parse JSON response
            analysis = self._parse_analysis_response(response.text)

            # Add computed fields
            analysis["aspect_ratio"] = aspect_ratio
            analysis["dimensions"] = reference_image.size

            # Print summary
            self._print_analysis_summary(analysis)

            return analysis

        except Exception as e:
            print(f"ERROR during template analysis: {e}")
            print("Falling back to basic analysis")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(reference_image)

    def generate_template_matching_prompt(self,
                                         analysis: Dict,
                                         user_subject: str) -> str:
        """
        Generate a detailed prompt that matches template structure with user's subject.

        This is CRITICAL for good baselines. Instead of just using the user's subject
        (e.g., "a young man"), we create a detailed prompt that preserves the template's
        visual structure.

        Args:
            analysis: Template analysis from analyze_template()
            user_subject: User's subject/content (e.g., "a young man", "a cat")

        Returns:
            Detailed template-matching prompt string

        Example:
            Input: analysis={lighting: "split lighting orange-blue"}, user="a young man"
            Output: "a young man, portrait photography, split lighting with orange and blue
                     color gels, dark background, professional studio lighting, f/2.8"
        """
        print("\n" + "="*70)
        print("GENERATING TEMPLATE-MATCHING PROMPT")
        print("="*70)
        print(f"User subject: '{user_subject}'")

        if not self.use_llm or self.model is None:
            # Fallback: basic combination
            return f"{user_subject}, professional photography, studio lighting"

        try:
            # Construct prompt for LLM
            prompt_text = f"""You are a text-to-image prompt engineer. Your task is to create a detailed prompt that will generate an image matching this template's visual structure but with a different subject.

**TEMPLATE ANALYSIS:**
```json
{json.dumps(analysis, indent=2)}
```

**USER'S SUBJECT:** "{user_subject}"

**YOUR TASK:**
Create a single, detailed text-to-image prompt that:

1. **Starts with the user's subject**: "{user_subject}"
2. **Preserves EXACT lighting** from template:
   - Include specific lighting type (e.g., "split lighting", "Rembrandt lighting")
   - Include light colors if present (e.g., "orange and blue color gels")
   - Include light direction and quality
3. **Preserves EXACT composition**:
   - Shot type (close-up, medium shot, etc.)
   - Camera angle
   - Framing approach
4. **Preserves EXACT background**:
   - Background type and color
   - Treatment (blurred, dark, gradient, etc.)
5. **Preserves style and mood**:
   - Photography style
   - Aesthetic qualities
6. **Include technical details** if evident:
   - Aperture hints (f/1.4 for bokeh, etc.)
   - Quality indicators

**OUTPUT REQUIREMENTS:**
- Single prompt string (NOT JSON)
- Start with user's subject
- Include all critical template elements
- Be specific (e.g., "orange and blue split lighting" not just "colored lighting")
- Professional photography terminology
- 40-80 words total

**EXAMPLE FORMAT:**
"{user_subject}, portrait photography, split lighting with orange and blue color gels, dark gradient background, professional studio setup, shallow depth of field, f/2.8, cinematic"

**OUTPUT ONLY THE PROMPT STRING. NO EXPLANATIONS. NO JSON. JUST THE PROMPT.**"""

            # Call Gemini
            response = self.model.generate_content(
                prompt_text,
                generation_config={"temperature": 0.2}
            )

            template_prompt = response.text.strip()

            # Clean up (remove quotes if LLM added them)
            if template_prompt.startswith('"') and template_prompt.endswith('"'):
                template_prompt = template_prompt[1:-1]
            if template_prompt.startswith("'") and template_prompt.endswith("'"):
                template_prompt = template_prompt[1:-1]

            print(f"\nGenerated template-matching prompt:")
            print(f"→ {template_prompt}")
            print()

            return template_prompt

        except Exception as e:
            print(f"ERROR generating template-matching prompt: {e}")
            print("Using fallback prompt")
            import traceback
            traceback.print_exc()

            # Fallback
            return f"{user_subject}, professional photography, studio lighting"

    def generate_specialized_vocabulary(self,
                                       analysis: Dict,
                                       size: int = 1000) -> Dict[str, List[str]]:
        """
        Generate vocabulary specialized for this template's characteristics.

        Unlike generic vocabularies, this focuses on modifiers relevant to the
        template's specific style (e.g., more lighting terms if template has
        complex lighting setup).

        Args:
            analysis: Template analysis
            size: Target total vocabulary size

        Returns:
            Dictionary with blocks: composition, lighting, style, quality, negative
        """
        print("\n" + "="*70)
        print("GENERATING SPECIALIZED VOCABULARY")
        print("="*70)
        print(f"Target size: {size} modifiers")

        if not self.use_llm or self.model is None:
            print("Using fallback vocabulary (LLM unavailable)")
            return self._fallback_vocabulary()

        try:
            # Construct vocabulary generation prompt
            prompt_text = f"""You are generating a specialized vocabulary for evolutionary prompt optimization.

**TEMPLATE CHARACTERISTICS:**
```json
{json.dumps(analysis, indent=2)}
```

**TASK:**
Generate approximately {size} text-to-image modifiers specifically useful for recreating images similar to this template.

**DISTRIBUTION STRATEGY:**

Based on template characteristics, allocate modifiers:

1. **lighting** (~35% = ~{int(size * 0.35)} terms):
   - Prioritize terms related to template's lighting style
   - If template has split lighting → include "split lighting", "two-tone lighting", "color gel lighting"
   - If template has specific colors → include those color names + "color gel", "colored lighting"
   - Include lighting directions, qualities, techniques
   - Examples: "Rembrandt lighting", "rim lighting", "orange side light", "blue backlight"

2. **composition** (~25% = ~{int(size * 0.25)} terms):
   - Match template's framing and perspective
   - If template is portrait → focus on portrait terms
   - If template is close-up → include "close-up", "extreme close-up", "headshot"
   - Include camera angles, focal lengths, framing approaches
   - Examples: "centered portrait", "rule of thirds", "shallow depth of field"

3. **style** (~20% = ~{int(size * 0.20)} terms):
   - Match template's aesthetic and mood
   - Photography styles, artistic approaches
   - Color grading terms
   - Examples: "cinematic", "editorial", "dramatic", "moody"

4. **quality** (~15% = ~{int(size * 0.15)} terms):
   - Technical quality indicators
   - Examples: "8k", "highly detailed", "sharp focus", "professional"

5. **negative** (~5% = ~{int(size * 0.05)} terms):
   - What to avoid
   - Examples: "blurry", "distorted", "low quality", "bad lighting"

**OUTPUT FORMAT (JSON):**
```json
{{
  "composition": ["term1", "term2", "term3", ...],
  "lighting": ["term1", "term2", "term3", ...],
  "style": ["term1", "term2", "term3", ...],
  "quality": ["term1", "term2", "term3", ...],
  "negative": ["term1", "term2", "term3", ...]
}}
```

**REQUIREMENTS:**
- Each term should be 1-4 words
- Include both general terms and highly specific ones
- Focus on terms relevant to THIS template's style
- Avoid redundancy (don't repeat same terms)
- Mix technical and natural language

**OUTPUT ONLY VALID JSON. NO MARKDOWN CODE FENCES. NO EXPLANATIONS.**"""

            # Call Gemini
            print("Generating vocabulary with Gemini...")
            response = self.model.generate_content(
                prompt_text,
                generation_config={"temperature": 0.3}
            )

            # Parse response
            vocab_data = self._parse_vocabulary_response(response.text)

            # Print statistics
            total = sum(len(v) for v in vocab_data.values())
            print(f"\nGenerated specialized vocabulary:")
            print(f"  composition: {len(vocab_data.get('composition', []))} terms")
            print(f"  lighting:    {len(vocab_data.get('lighting', []))} terms")
            print(f"  style:       {len(vocab_data.get('style', []))} terms")
            print(f"  quality:     {len(vocab_data.get('quality', []))} terms")
            print(f"  negative:    {len(vocab_data.get('negative', []))} terms")
            print(f"  TOTAL:       {total} terms")
            print()

            return vocab_data

        except Exception as e:
            print(f"ERROR generating specialized vocabulary: {e}")
            print("Using fallback vocabulary")
            import traceback
            traceback.print_exc()
            return self._fallback_vocabulary()

    # ==================== PRIVATE HELPER METHODS ====================

    def _construct_analysis_prompt(self) -> str:
        """Construct detailed template analysis prompt for Gemini."""
        return """Analyze this image in EXTREME detail for text-to-image prompt engineering purposes.

**CRITICAL FOCUS: LIGHTING ANALYSIS** (Most Important)

Describe the lighting with maximum specificity:
- **Lighting type**: Split lighting? Rembrandt? Butterfly? Loop? Broad? Short? Rim?
- **Light direction**: Front? Side? Back? Top? Bottom? 45-degree angle?
- **Light quality**: Hard shadows or soft diffused? Point source or area light?
- **Light colors**: ANY colored lighting? Gels? What specific colors? (e.g., "orange from left, blue from right")
- **Number of lights**: Key light only? Key + fill? Three-point? Multiple sources?
- **Special techniques**: Color gels? Practical lights? Motivated lighting?
- **Shadow characteristics**: Hard edges or soft? What direction? How intense?

**COMPOSITION ANALYSIS:**
- Shot type: Extreme close-up? Close-up? Medium close-up? Medium shot? Full shot?
- Camera angle: Eye level? Low angle? High angle? Dutch angle? Bird's eye?
- Framing: Centered? Rule of thirds? Golden ratio? Symmetrical? Off-balance?
- Focal length impression: Wide angle (<35mm)? Normal (50mm)? Telephoto (>85mm)?
- Depth of field: Shallow (bokeh, f/1.4-2.8)? Medium? Deep (f/8+)?
- Perspective: One-point? Two-point? Isometric? Forced perspective?

**BACKGROUND ANALYSIS:**
- Type: Solid color? Gradient? Textured? Environmental? Studio?
- Specific color: What exact color? (e.g., "dark charcoal gray", not just "dark")
- Treatment: Blurred? Sharp? Vignette? Gradient? Pattern?
- Complexity: Minimal? Clean? Busy? Detailed?

**STYLE & MOOD:**
- Photography type: Portrait? Editorial? Commercial? Fashion? Lifestyle? Documentary?
- Mood: Dramatic? Calm? Energetic? Mysterious? Intimate? Professional?
- Aesthetic: Cinematic? Minimalist? Maximalist? Vintage? Modern? Retro?
- Color grading: High contrast? Low contrast? Warm? Cool? Desaturated? Vibrant?

**COLOR PALETTE:**
- List 3-5 dominant colors (be specific: "burnt orange" not just "orange")
- Overall temperature: Warm? Cool? Neutral?
- Saturation: Highly saturated? Muted? Desaturated?
- Color relationships: Complementary? Analogous? Monochromatic? Split-complementary?

**TECHNICAL SPECIFICATIONS:**
- Apparent aperture: f/1.4? f/2.8? f/5.6? f/8? (based on depth of field)
- Apparent ISO: Low (clean)? Medium? High (grainy)?
- Sharpness: Tack sharp? Soft? Motion blur?
- Quality level: Professional? Amateur? Phone camera?

**CRITICAL ELEMENTS** (List 5-10):
What are the MOST IMPORTANT visual characteristics that MUST be preserved to recreate this image's look?
Priority order: Lighting > Composition > Background > Style

**OUTPUT FORMAT (JSON):**
```json
{
  "lighting": {
    "type": "specific lighting technique name",
    "direction": "detailed light direction",
    "quality": "hard/soft with details",
    "colors": ["specific color 1", "specific color 2"] or [] if no colored lighting,
    "num_sources": "estimated number of light sources",
    "description": "comprehensive 2-3 sentence lighting description"
  },
  "composition": {
    "shot_type": "specific shot type",
    "angle": "camera angle",
    "framing": "framing approach",
    "focal_length": "apparent focal length category",
    "depth_of_field": "shallow/medium/deep with f-stop estimate"
  },
  "background": {
    "type": "background type",
    "color": "specific color name",
    "treatment": "how background is treated",
    "description": "brief background description"
  },
  "style": {
    "photography_type": "specific photography genre",
    "mood": "mood description",
    "aesthetic": "aesthetic description",
    "color_grading": "color grading approach"
  },
  "color_palette": ["specific color 1", "specific color 2", "specific color 3"],
  "technical": {
    "aperture": "f-stop estimate",
    "sharpness": "sharpness assessment",
    "quality_level": "quality assessment"
  },
  "critical_elements": [
    "First most critical element (usually lighting)",
    "Second most critical element",
    "Third most critical element",
    "etc (5-10 total)"
  ],
  "template_description": "One comprehensive sentence (40-60 words) describing this image's complete visual structure"
}
```

**CRITICAL REQUIREMENTS:**
- Be EXTREMELY specific about lighting (this is the #1 most important aspect)
- Use professional photography terminology
- Output ONLY valid JSON
- No markdown code fences
- No explanatory text outside the JSON"""

    def _detect_aspect_ratio(self, image: Image.Image) -> str:
        """
        Detect aspect ratio category.

        Returns: "portrait", "landscape", or "square"
        """
        width, height = image.size
        ratio = width / height

        if ratio > 1.2:
            return "landscape"
        elif ratio < 0.8:
            return "portrait"
        else:
            return "square"

    def _image_to_part(self, image: Image.Image) -> Part:
        """Convert PIL Image to Vertex AI Part for Gemini."""
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Create Part
        return Part.from_data(image_bytes, mime_type="image/jpeg")

    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse JSON analysis response from Gemini."""
        # Clean response text
        cleaned = response_text.strip()

        # Remove markdown code fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Parse JSON
        try:
            analysis = json.loads(cleaned)
            return analysis
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response text (first 500 chars): {response_text[:500]}")
            raise

    def _parse_vocabulary_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse vocabulary JSON response from Gemini."""
        # Clean response
        cleaned = response_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Parse JSON
        try:
            vocab_data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Failed to parse vocabulary JSON: {e}")
            print(f"Response (first 300 chars): {response_text[:300]}")
            raise

        # Ensure all required blocks are present
        required_blocks = ["composition", "lighting", "style", "quality", "negative"]
        for block in required_blocks:
            if block not in vocab_data:
                print(f"Warning: Missing block '{block}', adding empty list")
                vocab_data[block] = []

        return vocab_data

    def _fallback_analysis(self, image: Image.Image) -> Dict:
        """Fallback analysis when LLM is unavailable."""
        return {
            "aspect_ratio": self._detect_aspect_ratio(image),
            "dimensions": image.size,
            "lighting": {
                "type": "unknown",
                "description": "No LLM available for analysis"
            },
            "composition": {
                "shot_type": "unknown"
            },
            "background": {
                "type": "unknown"
            },
            "style": {
                "photography_type": "portrait"
            },
            "color_palette": [],
            "technical": {},
            "critical_elements": ["lighting", "composition"],
            "template_description": "Reference template image"
        }

    def _fallback_vocabulary(self) -> Dict[str, List[str]]:
        """Fallback vocabulary when LLM is unavailable."""
        return {
            "composition": [
                "portrait", "centered", "close-up", "medium shot",
                "rule of thirds", "shallow depth of field"
            ],
            "lighting": [
                "studio lighting", "professional lighting", "soft lighting",
                "dramatic lighting", "natural lighting"
            ],
            "style": [
                "photorealistic", "professional", "cinematic",
                "high quality", "editorial"
            ],
            "quality": [
                "8k", "high resolution", "highly detailed",
                "sharp focus", "professional photography"
            ],
            "negative": [
                "blurry", "low quality", "distorted",
                "bad lighting", "amateur"
            ]
        }

    def _print_analysis_summary(self, analysis: Dict):
        """Print human-readable analysis summary."""
        print(f"\n{'='*70}")
        print("TEMPLATE ANALYSIS SUMMARY")
        print(f"{'='*70}")

        print(f"\nImage Properties:")
        print(f"  Aspect Ratio: {analysis.get('aspect_ratio', 'N/A')}")
        print(f"  Dimensions: {analysis.get('dimensions', 'N/A')}")

        if "lighting" in analysis:
            lighting = analysis["lighting"]
            print(f"\nLighting:")
            print(f"  Type: {lighting.get('type', 'N/A')}")
            print(f"  Colors: {', '.join(lighting.get('colors', [])) or 'None'}")
            print(f"  Description: {lighting.get('description', 'N/A')}")

        if "composition" in analysis:
            comp = analysis["composition"]
            print(f"\nComposition:")
            print(f"  Shot Type: {comp.get('shot_type', 'N/A')}")
            print(f"  Angle: {comp.get('angle', 'N/A')}")
            print(f"  Depth of Field: {comp.get('depth_of_field', 'N/A')}")

        if "style" in analysis:
            style = analysis["style"]
            print(f"\nStyle:")
            print(f"  Type: {style.get('photography_type', 'N/A')}")
            print(f"  Mood: {style.get('mood', 'N/A')}")

        if "critical_elements" in analysis and analysis["critical_elements"]:
            print(f"\nCritical Elements (MUST preserve):")
            for i, elem in enumerate(analysis["critical_elements"][:5], 1):
                print(f"  {i}. {elem}")

        if "template_description" in analysis:
            print(f"\nTemplate Description:")
            print(f"  {analysis['template_description']}")

        print(f"{'='*70}\n")
