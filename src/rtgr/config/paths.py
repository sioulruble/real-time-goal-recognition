from pathlib import Path
import os

# Project root (two levels up from src/config)
ROOT = Path(__file__).resolve().parents[3]

# Common folders (can be overridden by env vars)
SRC_DIR = ROOT / "src"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

# Inference / artifacts used by the codebase
YOLO_MODEL = Path(os.getenv("YOLO_MODEL", MODELS_DIR / "yolo26s-seg.pt"))
TRANSITION_MATRIX = Path(os.getenv("TRANSITION_MATRIX", DATA_DIR / "transition_proba_for_hmm.json"))
GOALS_TYPE_CSV = Path(os.getenv("GOALS_TYPE_CSV", DATA_DIR / "goals_type.csv"))
TIME_SPENT_CSV = Path(os.getenv("TIME_SPENT_CSV", DATA_DIR / "time_spent.csv"))
RELATED_GOALS_JSON = Path(os.getenv("RELATED_GOALS_JSON", DATA_DIR / "related_goals.json"))
# Camera / recording defaults
DEFAULT_SVO = Path(os.getenv("DEFAULT_SVO", ROOT / "recorded_stream.svo"))

def ensure_dirs(*paths):
    """Create directories if they do not exist. Returns list of Path objects."""
    created = []
    for p in paths:
        p = Path(p)
        if p.suffix:  # likely a file path -> create parent
            p = p.parent
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
    return created

def resolve(p):
    """Resolve a path-like to absolute Path under project root when relative."""
    p = Path(p)
    if not p.is_absolute():
        return (ROOT / p).resolve()
    return p.resolve()

# Ensure common dirs exist on import
ensure_dirs(DATA_DIR, MODELS_DIR, SRC_DIR)