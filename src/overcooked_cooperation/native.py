"""Load the bundled native transitions and check their source identity."""

import sys
from hashlib import sha256
from pathlib import Path

from .io import read_json

ROOT = Path(__file__).resolve().parent


def load_engine():
    vendor = ROOT / "_vendor"
    package = vendor / "overcooked_ai_py"
    for name, digest in read_json(ROOT / "provenance.json")["native_engine"][
        "files_sha256"
    ].items():
        if sha256((package / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned native source differs: " + name)
    loaded = sys.modules.get("overcooked_ai_py")
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        raise RuntimeError("Another overcooked_ai_py engine is already loaded")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    return OvercookedGridworld
