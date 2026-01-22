#!/bin/bash
# Codespaces installation troubleshooting script

echo "=== BTC Engine Installation Diagnostics ==="
echo ""

# Check Python version
echo "1. Python version:"
python --version
echo ""

# Check if files exist
echo "2. Checking critical files:"
for file in "pyproject.toml" "setup.py" "src/btc_engine/__init__.py"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file MISSING"
    fi
done
echo ""

# Check pyproject.toml build backend
echo "3. Build backend in pyproject.toml:"
grep "build-backend" pyproject.toml || echo "  ✗ build-backend not found"
echo ""

# Check git status
echo "4. Git status:"
git status --short
echo ""

# Try to get detailed error
echo "5. Attempting installation with verbose output:"
pip install --verbose -e . 2>&1 | tail -50

echo ""
echo "=== Diagnostics Complete ==="
