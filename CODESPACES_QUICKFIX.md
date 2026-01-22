# Quick Fix for Codespaces Installation

## In your Codespace, run these commands:

### Step 1: Verify files are updated
```bash
# Pull latest changes
git pull

# Check if pyproject.toml was updated
grep "setuptools" pyproject.toml
# Should show: requires = ["setuptools>=45", "wheel", "setuptools-scm>=6.2"]

# Check if setup.py exists
ls -la setup.py
```

### Step 2: Clean install
```bash
# Clean any cached build files
rm -rf build/ dist/ *.egg-info/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install
pip install -e .
```

### Step 3: If still failing, use alternative method
```bash
# Install dependencies directly
pip install numpy pandas scipy scikit-learn lightgbm statsmodels \
    duckdb sqlalchemy httpx aiohttp requests pydantic pydantic-settings \
    typer prefect rich plotly matplotlib seaborn streamlit \
    python-dotenv PyYAML jinja2 pyarrow fastparquet tenacity cachetools tqdm

# Then install package without editable mode
pip install .
```

### Step 4: Run diagnostics (if needed)
```bash
chmod +x diagnose.sh
./diagnose.sh
```

## Quick Test
```bash
# After installation
python -c "import btc_engine; print('✓ Import successful')"
btc-engine --help
```

## If nothing works - Manual setup
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/workspaces/btc-inference-engine/src"

# Run CLI directly
python -m btc_engine.cli --help
```

## Common Issues

**Issue**: "metadata-generation-failed"
**Cause**: Build configuration conflict or missing dependencies
**Fix**: Use Step 3 (non-editable install) or manual setup

**Issue**: Files not updated after git pull
**Cause**: Codespace using cached version
**Fix**: 
```bash
# Hard reset to latest
git fetch origin
git reset --hard origin/main
```

**Issue**: Wrong Python version  
**Fix**: Rebuild devcontainer (Cmd/Ctrl+Shift+P → "Rebuild Container")
