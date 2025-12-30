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
    Create a curated vocabulary of style modifiers for prompt enhancement

    Returns:
        List of modifier strings categorized by type
    """
    modifiers = [
        # Quality modifiers
        "8k", "4k", "high resolution", "highly detailed", "sharp focus",
        "ultra detailed", "professional", "masterpiece", "best quality",

        # Style modifiers
        "cinematic", "photorealistic", "digital art", "illustration",
        "oil painting", "watercolor", "concept art", "fantasy art",
        "anime style", "studio ghibli style", "pixar style",

        # Lighting modifiers
        "natural lighting", "soft lighting", "golden hour", "volumetric lighting",
        "dramatic lighting", "studio lighting", "rim lighting", "ambient lighting",
        "neon lighting", "moody lighting", "backlit", "side lighting",

        # Composition modifiers
        "wide angle", "close-up", "portrait", "aerial view", "bird's eye view",
        "low angle", "high angle", "centered composition", "rule of thirds",

        # Technical modifiers
        "depth of field", "bokeh", "HDR", "long exposure", "macro photography",
        "tilt-shift", "motion blur", "lens flare",

        # Atmosphere modifiers
        "moody", "atmospheric", "ethereal", "dreamy", "vibrant colors",
        "muted colors", "warm tones", "cool tones", "saturated",
    ]

    return modifiers


def create_negative_vocab() -> List[str]:
    """
    Create a vocabulary of negative prompt terms

    Returns:
        List of negative modifier strings
    """
    negative = [
        "blurry", "distorted", "low quality", "bad quality", "worst quality",
        "low resolution", "pixelated", "artifacts", "noise", "grainy",
        "oversaturated", "undersaturated", "overexposed", "underexposed",
        "ugly", "deformed", "disfigured", "bad anatomy", "bad proportions",
        "watermark", "text", "signature", "cropped", "out of frame",
    ]

    return negative


def create_block_vocabularies() -> Dict[str, List[str]]:
    """
    Create structured vocabularies for block-based genomes (Experiment 2)

    Returns:
        Dictionary with vocabulary lists for each block type
    """
    vocabularies = {
        "composition": [
            "wide angle", "close-up", "portrait", "aerial view", "bird's eye view",
            "low angle", "high angle", "centered composition", "rule of thirds",
            "symmetrical", "asymmetrical", "full body", "medium shot", "extreme close-up",
            "over the shoulder", "dutch angle", "panoramic", "macro",
        ],

        "lighting": [
            "natural lighting", "soft lighting", "golden hour", "volumetric lighting",
            "dramatic lighting", "studio lighting", "rim lighting", "ambient lighting",
            "neon lighting", "moody lighting", "backlit", "side lighting",
            "harsh lighting", "diffused lighting", "sunset", "sunrise", "blue hour",
            "candlelight", "moonlight", "overcast", "chiaroscuro",
        ],

        "style": [
            "photorealistic", "cinematic", "digital art", "illustration",
            "oil painting", "watercolor", "concept art", "fantasy art",
            "anime style", "studio ghibli style", "pixar style", "3d render",
            "line art", "sketch", "charcoal", "pastel", "impressionist",
            "surreal", "abstract", "minimalist", "maximalist",
        ],

        "quality": [
            "8k", "4k", "high resolution", "highly detailed", "sharp focus",
            "ultra detailed", "professional", "masterpiece", "best quality",
            "HDR", "depth of field", "bokeh", "crisp", "crystal clear",
        ],

        "negative": [
            "blurry", "distorted", "low quality", "bad quality", "worst quality",
            "low resolution", "pixelated", "artifacts", "noise", "grainy",
            "oversaturated", "undersaturated", "overexposed", "underexposed",
            "ugly", "deformed", "disfigured", "bad anatomy", "bad proportions",
            "watermark", "text", "signature", "cropped", "out of frame",
        ]
    }

    return vocabularies
