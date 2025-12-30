"""
Utility functions for the evolutionary prompt engineering project
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging
from pathlib import Path


class Logger:
    """Simple file + console logger"""

    def __init__(self, log_dir: str = "logs", name: str = "evolution"):
        """
        Initialize logger

        Args:
            log_dir: Directory to save log files
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        # Configure logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


def load_json(file_path: str) -> Any:
    """
    Load JSON data from file

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_results(results: Dict[str, Any], experiment_name: str, output_dir: str = "data/results"):
    """
    Save experiment results to JSON file with timestamp

    Args:
        results: Results dictionary
        experiment_name: Name of the experiment
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    save_json(results, file_path)
    return file_path


def create_modifier_vocab() -> List[str]:
    """
    Create an expanded vocabulary of style modifiers for prompt enhancement

    Returns:
        List of modifier strings categorized by type
    """
    modifiers = [
        # Quality modifiers (expanded)
        "8k", "4k", "16k", "high resolution", "highly detailed", "sharp focus",
        "ultra detailed", "professional", "masterpiece", "best quality",
        "ultra high quality", "extremely detailed", "intricate details",
        "fine details", "hyper detailed", "ultra sharp", "crisp details",
        "award winning", "trending on artstation", "featured on behance",

        # Style modifiers (expanded)
        "cinematic", "photorealistic", "digital art", "illustration",
        "oil painting", "watercolor", "concept art", "fantasy art",
        "anime style", "studio ghibli style", "pixar style", "disney style",
        "hyperrealistic", "surrealism", "impressionism", "expressionism",
        "pop art", "art nouveau", "art deco", "baroque", "renaissance",
        "gothic", "steampunk", "cyberpunk", "vaporwave", "synthwave",
        "retro", "vintage", "modern", "contemporary", "futuristic",
        "minimalist", "maximalist", "abstract", "geometric",
        "3d render", "cgi", "octane render", "unreal engine", "blender",
        "vector art", "line art", "sketch", "charcoal drawing", "pencil sketch",
        "manga style", "comic book style", "graphic novel", "storybook illustration",

        # Lighting modifiers (expanded)
        "natural lighting", "soft lighting", "golden hour", "volumetric lighting",
        "dramatic lighting", "studio lighting", "rim lighting", "ambient lighting",
        "neon lighting", "moody lighting", "backlit", "side lighting",
        "cinematic lighting", "high key lighting", "low key lighting",
        "rembrandt lighting", "butterfly lighting", "split lighting",
        "hard lighting", "diffused light", "specular highlights",
        "god rays", "light rays", "sun rays", "lens flare",
        "blue hour", "sunset lighting", "sunrise lighting", "twilight",
        "candlelight", "firelight", "moonlight", "starlight",
        "fluorescent", "incandescent", "LED lighting", "stage lighting",
        "spotlight", "fill light", "key light", "hair light",

        # Composition modifiers (expanded)
        "wide angle", "close-up", "portrait", "aerial view", "bird's eye view",
        "low angle", "high angle", "centered composition", "rule of thirds",
        "symmetrical", "asymmetrical", "dynamic composition", "balanced composition",
        "diagonal composition", "leading lines", "framing", "negative space",
        "full body shot", "medium shot", "extreme close-up", "establishing shot",
        "over the shoulder", "dutch angle", "panoramic", "macro",
        "fisheye", "telephoto", "35mm", "50mm", "85mm lens",
        "eye level", "worm's eye view", "overhead shot",

        # Technical modifiers (expanded)
        "depth of field", "shallow depth of field", "deep depth of field",
        "bokeh", "HDR", "long exposure", "macro photography",
        "tilt-shift", "motion blur", "lens flare", "chromatic aberration",
        "film grain", "vignette", "ray tracing", "global illumination",
        "subsurface scattering", "ambient occlusion", "anti-aliasing",
        "f/1.4", "f/2.8", "f/8", "ISO 100", "1/500 shutter speed",

        # Atmosphere modifiers (expanded)
        "moody", "atmospheric", "ethereal", "dreamy", "vibrant colors",
        "muted colors", "warm tones", "cool tones", "saturated", "desaturated",
        "high contrast", "low contrast", "vivid", "pastel colors",
        "monochromatic", "sepia", "black and white", "duotone",
        "rich colors", "bold colors", "subtle colors", "earthy tones",
        "neon colors", "candy colors", "jewel tones", "metallic",
        "misty", "foggy", "hazy", "smoky", "dusty",
        "rainy", "snowy", "sunny", "cloudy", "stormy",
        "magical", "mystical", "enchanting", "whimsical", "romantic",
        "epic", "grand", "majestic", "serene", "peaceful", "tranquil",
        "dark", "light", "bright", "dim", "glowing",

        # Subject enhancement modifiers
        "elegant", "luxurious", "premium", "high-end", "sophisticated",
        "polished", "refined", "sleek", "modern design", "classic design",
        "detailed texture", "realistic texture", "smooth surface", "glossy",
        "matte finish", "reflective", "transparent", "translucent",
    ]

    return modifiers


def create_negative_vocab() -> List[str]:
    """
    Create an expanded vocabulary of negative prompt terms

    Returns:
        List of negative modifier strings
    """
    negative = [
        # Quality issues
        "blurry", "distorted", "low quality", "bad quality", "worst quality",
        "low resolution", "pixelated", "artifacts", "noise", "grainy",
        "jpeg artifacts", "compression artifacts", "aliasing", "banding",
        "fuzzy", "soft focus", "out of focus", "unfocused",

        # Exposure issues
        "oversaturated", "undersaturated", "overexposed", "underexposed",
        "too dark", "too bright", "washed out", "muddy colors",
        "blown highlights", "crushed blacks", "flat lighting",

        # Composition issues
        "cropped", "out of frame", "cut off", "poorly framed",
        "tilted", "unbalanced", "cluttered", "messy background",
        "distracting background", "busy background",

        # Anatomical/structural issues
        "ugly", "deformed", "disfigured", "bad anatomy", "bad proportions",
        "malformed", "mutated", "extra limbs", "missing limbs",
        "fused fingers", "too many fingers", "long neck", "distorted face",

        # Unwanted elements
        "watermark", "text", "signature", "logo", "copyright",
        "username", "artist name", "border", "frame", "timestamp",

        # Style issues
        "amateurish", "unprofessional", "cheap looking", "tacky",
        "cartoonish", "unrealistic", "fake looking", "plastic looking",

        # Technical issues
        "chromatic aberration", "lens distortion", "vignetting",
        "motion blur", "camera shake", "double exposure",
    ]

    return negative


def create_block_vocabularies() -> Dict[str, List[str]]:
    """
    Create expanded structured vocabularies for block-based genomes (Experiment 2)

    Returns:
        Dictionary with vocabulary lists for each block type
    """
    vocabularies = {
        "composition": [
            # Camera angles
            "wide angle", "close-up", "portrait", "aerial view", "bird's eye view",
            "low angle", "high angle", "eye level", "worm's eye view", "overhead shot",
            "dutch angle", "canted angle", "tilted frame",
            # Framing
            "centered composition", "rule of thirds", "golden ratio",
            "symmetrical", "asymmetrical", "balanced composition", "dynamic composition",
            "diagonal composition", "leading lines", "framing", "negative space",
            # Shot types
            "full body", "medium shot", "extreme close-up", "establishing shot",
            "over the shoulder", "panoramic", "macro", "product shot",
            "three-quarter view", "profile view", "front view", "back view",
            # Lens effects
            "fisheye", "telephoto", "35mm", "50mm", "85mm lens", "24mm wide",
            "tilt-shift", "selective focus", "deep focus",
        ],

        "lighting": [
            # Natural lighting
            "natural lighting", "soft lighting", "golden hour", "blue hour",
            "sunset lighting", "sunrise lighting", "twilight", "daylight",
            "overcast", "cloudy day", "bright sunny", "dappled light",
            # Studio lighting
            "studio lighting", "professional lighting", "product lighting",
            "high key lighting", "low key lighting", "three-point lighting",
            "rembrandt lighting", "butterfly lighting", "split lighting", "loop lighting",
            "key light", "fill light", "hair light", "background light",
            # Dramatic lighting
            "dramatic lighting", "volumetric lighting", "rim lighting", "backlit",
            "side lighting", "harsh lighting", "hard shadows", "soft shadows",
            "chiaroscuro", "silhouette", "contre-jour",
            # Atmospheric lighting
            "moody lighting", "ambient lighting", "diffused lighting", "diffused light",
            "god rays", "light rays", "sun rays", "lens flare",
            "specular highlights", "reflections", "caustics",
            # Artificial lighting
            "neon lighting", "candlelight", "firelight", "moonlight", "starlight",
            "fluorescent", "incandescent", "LED lighting", "stage lighting", "spotlight",
            "warm light", "cool light", "colored lighting", "mixed lighting",
        ],

        "style": [
            # Photographic styles
            "photorealistic", "hyperrealistic", "cinematic", "editorial",
            "fashion photography", "commercial photography", "product photography",
            "fine art photography", "documentary style", "journalistic",
            # Digital art styles
            "digital art", "digital painting", "3d render", "cgi",
            "octane render", "unreal engine", "blender render", "v-ray",
            # Traditional art styles
            "oil painting", "watercolor", "acrylic painting", "gouache",
            "pastel", "charcoal", "pencil sketch", "ink drawing",
            "impressionist", "expressionist", "surrealist", "cubist",
            # Illustration styles
            "illustration", "concept art", "fantasy art", "sci-fi art",
            "anime style", "manga style", "studio ghibli style", "pixar style",
            "disney style", "dreamworks style", "comic book style", "graphic novel",
            "storybook illustration", "children's book", "whimsical illustration",
            # Design styles
            "minimalist", "maximalist", "abstract", "geometric",
            "art nouveau", "art deco", "baroque", "renaissance",
            "gothic", "victorian", "retro", "vintage", "modern", "contemporary",
            "futuristic", "steampunk", "cyberpunk", "vaporwave", "synthwave",
            # Texture and finish
            "matte", "glossy", "textured", "smooth", "rough",
            "metallic", "glass-like", "crystal", "holographic",
        ],

        "quality": [
            # Resolution
            "8k", "4k", "16k", "high resolution", "ultra high resolution",
            "extremely detailed", "highly detailed", "intricate details",
            "fine details", "hyper detailed", "ultra detailed",
            # Sharpness
            "sharp focus", "ultra sharp", "crisp", "crystal clear", "tack sharp",
            "razor sharp", "pin sharp",
            # Professional quality
            "professional", "masterpiece", "best quality", "award winning",
            "trending on artstation", "featured on behance",
            "museum quality", "gallery worthy", "exhibition quality",
            # Technical quality
            "HDR", "depth of field", "bokeh", "ray traced",
            "global illumination", "subsurface scattering",
            "anti-aliased", "clean render", "polished",
            # Production quality
            "commercial quality", "print quality", "publication ready",
            "high fidelity", "pristine", "flawless", "immaculate",
        ],

        "negative": [
            # Quality issues
            "blurry", "distorted", "low quality", "bad quality", "worst quality",
            "low resolution", "pixelated", "artifacts", "noise", "grainy",
            "jpeg artifacts", "compression artifacts", "aliasing", "banding",
            "fuzzy", "soft focus", "out of focus", "unfocused",
            # Exposure issues
            "oversaturated", "undersaturated", "overexposed", "underexposed",
            "too dark", "too bright", "washed out", "muddy colors",
            "blown highlights", "crushed blacks", "flat lighting",
            # Composition issues
            "cropped", "out of frame", "cut off", "poorly framed",
            "tilted", "unbalanced", "cluttered", "messy background",
            "distracting background", "busy background",
            # Structural issues
            "ugly", "deformed", "disfigured", "bad anatomy", "bad proportions",
            "malformed", "mutated", "distorted", "warped",
            # Unwanted elements
            "watermark", "text", "signature", "logo", "copyright",
            "username", "artist name", "border", "frame", "timestamp",
            # Style issues
            "amateurish", "unprofessional", "cheap looking", "tacky",
            "cartoonish", "unrealistic", "fake looking", "plastic looking",
        ]
    }

    return vocabularies
