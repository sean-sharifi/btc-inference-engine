# Installation Guide

## Issue Resolution

The package installation error has been fixed. Two changes were made:

1. **Added `setup.py`** - For compatibility with older pip versions
2. **Fixed `pyproject.toml`** - Added correct package path configuration

## Installation Instructions

### Option 1: Upgrade pip first (Recommended)

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Then install the package
pip install -e .
```

### Option 2: Install with current pip

```bash
# Direct installation (setup.py now exists)
pip install -e .
```

### Option 3: Use uv (Modern Package Manager)

```bash
# Install uv (if not already installed)
pip install uv

# Install with uv
uv pip install -e .
```

## Verification

After installation, verify it works:

```bash
# Check if command is available
btc-engine --help

# Should show all commands:
# - init-db
# - ingest-deribit
# - ingest-glassnode
# - build-features
# - train-model
# - forecast
# - evaluate
# - newsletter
# - dashboard
# - status
```

## Troubleshooting

If you still have issues:

1. **Check Python version**: `python --version` (need 3.11+)
2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

3. **Install dependencies manually**:
   ```bash
   pip install -r requirements.txt  # If we create this
   ```

## What Was Fixed

- ❌ Before: Missing `setup.py`, incompatible with pip < 21.3
- ✅ After: `setup.py` added, works with all pip versions
- ✅ `pyproject.toml` corrected with proper package path

The system is now ready for installation!
