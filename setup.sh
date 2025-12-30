#!/bin/bash

# Setup script for Evolutionary Prompt Engineering project

echo "=================================="
echo "Evolutionary Prompt Engineering"
echo "Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if not exists
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo "   - FAL_KEY: Required for image generation"
    echo "   - GOOGLE_CLOUD_PROJECT: Optional for LLM features"
    echo ""
else
    echo "⚠️  .env file already exists"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/results
mkdir -p data/reference_templates
mkdir -p logs
echo "✓ Directories created"
echo ""

# Run tests
echo "=================================="
echo "Running Tests"
echo "=================================="
echo ""

read -p "Do you want to run tests now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running model tests..."
    python tests/test_models.py
    echo ""

    echo "Running fitness tests..."
    python tests/test_fitness.py
    echo ""

    echo "Running LLM generator tests..."
    python tests/test_llm_generator.py
    echo ""
fi

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Start Jupyter: jupyter notebook"
echo "4. Open: experiments/01_prompt_enhancement.ipynb"
echo ""
echo "For help, see README.md"
echo ""
