"""Project-wide constants and paths"""

from pathlib import Path

# Project root (go up 3 levels from this file: utils -> btc_engine -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Configuration
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Date formats
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Asset
ASSET = "BTC"
ASSET_PAIR = "BTC-USD"

# Options parameters
MIN_IV = 0.01
MAX_IV = 5.0
MIN_OI = 0.1

# Numerical constants
EPSILON = 1e-10
