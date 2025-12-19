#!/bin/bash
# Quick setup script for OC6 environment (conda + uv hybrid)

set -e  # Exit on error

echo "================================================"
echo "OC6 Environment Setup (Conda + uv hybrid)"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Anaconda or Miniconda first."
    echo "   Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^oc6 "; then
    echo "⚠️  Environment 'oc6' already exists."
    read -p "   Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        conda env remove -n oc6 -y
    else
        echo "   Skipping environment creation."
        echo "   To update: conda env update -f environment.yml"
        exit 0
    fi
fi

# Create conda environment
echo "📦 Creating conda environment 'oc6'..."
echo "   This installs: Python 3.11, NumPy, pandas, scikit-learn, Jupyter, etc."
conda env create -f environment.yml

echo ""
echo "✓ Conda environment created!"
echo ""

# Activate environment and install uv packages
echo "📦 Installing specialized packages with uv..."
echo "   This installs: MLflow, XGBoost, FastAPI, etc."

# Use conda run to execute in the oc6 environment
conda run -n oc6 uv pip install -e ".[dev]"

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   conda activate oc6"
echo ""
echo "2. Verify installation:"
echo "   python -c 'import numpy, pandas, mlflow; print(\"✓ All working!\")'"
echo ""
echo "3. Start working:"
echo "   jupyter lab                 # Launch notebooks"
echo "   mlflow ui --port 5000       # Start MLflow UI"
echo ""
echo "📚 See SETUP_ENVIRONMENT.md for detailed documentation"
echo ""
