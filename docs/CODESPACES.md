# GitHub Codespaces Setup Guide

## Issue
Codespaces uses Python 3.7 by default, but this project requires Python 3.11+.

## Solution
I've created a `.devcontainer/devcontainer.json` configuration that will automatically set up Python 3.11 in Codespaces.

## How to Use

### Option 1: Rebuild Container (Recommended)
1. In Codespaces, press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Rebuild Container"
3. Select "Codespaces: Rebuild Container"
4. Wait for the rebuild (3-5 minutes)
5. The project will auto-install with Python 3.11

### Option 2: Create New Codespace
1. Delete current codespace
2. Create new codespace from your repo
3. The `.devcontainer` config will automatically set up Python 3.11
4. Project will auto-install via `postCreateCommand`

### Option 3: Manual Setup (If devcontainer doesn't work)
```bash
# Install Python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project
pip install -e .
```

## Verify Installation

After setup:
```bash
# Check Python version
python --version  # Should show Python 3.11.x

# Check installation
btc-engine --help  # Should show all commands

# Check dashboard port
# Port 8501 is auto-forwarded for Streamlit dashboard
```

## What the Devcontainer Does

1. **Sets Python 3.11** as the default interpreter
2. **Auto-installs** the project on container creation
3. **Forwards port 8501** for the dashboard
4. **Configures VS Code** with Python extensions
5. **Sets up environment** for immediate development

## Troubleshooting

**If rebuild fails:**
- Check `.devcontainer/devcontainer.json` is committed to repo
- Ensure you have permissions to rebuild container
- Try creating a fresh codespace

**If manual setup needed:**
```bash
# Verify Python
which python3.11

# If not found, install:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11

# Then follow manual setup steps above
```

## Configuration Files Added

- `.devcontainer/devcontainer.json` - Container configuration
- Commit and push this to your GitHub repo
- All future codespaces will use Python 3.11 automatically

## Quick Start After Setup

```bash
# Initialize database
btc-engine init-db

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run dashboard
btc-engine dashboard
# Access via forwarded port 8501
```
