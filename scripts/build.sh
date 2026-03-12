#!/bin/bash
# Build script for madakit distribution

set -e

echo "Building madakit distribution packages..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install build dependencies
pip install --upgrade build twine

# Build source distribution and wheel
python -m build

# Check distribution
twine check dist/*

echo ""
echo "Build complete! Distribution packages:"
ls -lh dist/

echo ""
echo "To publish to TestPyPI:"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To publish to PyPI:"
echo "  twine upload dist/*"
echo ""
echo "Or use GitHub Actions by pushing the v1.0.0 tag:"
echo "  git push origin v1.0.0"
