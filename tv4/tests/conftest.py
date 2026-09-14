from __future__ import annotations

import sys
from pathlib import Path

_TV4_SRC = Path(__file__).resolve().parents[1] / "src"
_WP08_SRC = Path(__file__).resolve().parents[2] / "TV2" / "WP08" / "src"

for p in (_TV4_SRC, _WP08_SRC):
    if p.exists():
        p_str = str(p.resolve())
        if p_str in sys.path:
            sys.path.remove(p_str)
        sys.path.insert(0, p_str)

# Purge any previously imported modules pointing to runtime tv4
for mod_name in list(sys.modules.keys()):
    if mod_name == "tv4" or mod_name.startswith("tv4."):
        sys.modules.pop(mod_name, None)
