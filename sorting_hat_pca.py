#!/usr/bin/env python3
"""
House primitive scores (7×15) and PCA via SVD.

House definitions are loaded from houses.json (same directory as this file).
The decomposition runs once when this module is first imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy is required:  pip install numpy")
    sys.exit(1)

# ── Maps `terminal_color` in houses.json → ANSI foreground (bars + titles) ──
TERMINAL_COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "green": "\033[92m",
    "cyan": "\033[96m",
    "blue": "\033[94m",
    "pink": "\033[95m",
}

_HOUSES_JSON = Path(__file__).resolve().parent / "houses.json"


def _load_houses(path: Path) -> tuple[list[str], dict[str, list[float]], list[str], list[dict]]:
    """Parse houses.json; return PRIM_NAMES, HOUSE_PRIM, HOUSE_ORDER, raw house dicts."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Cannot read {path}: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e

    axes = data.get("primitive_axes")
    if not isinstance(axes, list) or len(axes) != 15:
        raise RuntimeError(f"{path}: 'primitive_axes' must be a list of 15 names")

    houses = data.get("houses")
    if not isinstance(houses, list) or len(houses) < 2:
        raise RuntimeError(f"{path}: 'houses' must be a list with at least 2 entries")

    house_prim: dict[str, list[float]] = {}
    order: list[str] = []

    for i, h in enumerate(houses):
        if not isinstance(h, dict):
            raise RuntimeError(f"{path}: houses[{i}] must be an object")
        name = h.get("name")
        scores = h.get("primitive_scores")
        if not name or not isinstance(name, str):
            raise RuntimeError(f"{path}: houses[{i}].name must be a non-empty string")
        if not isinstance(scores, list) or len(scores) != 15:
            raise RuntimeError(
                f"{path}: houses[{i}].primitive_scores must be a list of 15 numbers "
                f"(got {len(scores) if isinstance(scores, list) else type(scores)})"
            )
        try:
            scores_f = [float(x) for x in scores]
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"{path}: houses[{i}].primitive_scores must be numbers") from e
        if name in house_prim:
            raise RuntimeError(f"{path}: duplicate house name {name!r}")
        col = h.get("terminal_color", "")
        if not isinstance(col, str) or col.lower() not in TERMINAL_COLORS:
            raise RuntimeError(
                f"{path}: houses[{i}].terminal_color must be one of: "
                f"{', '.join(sorted(TERMINAL_COLORS))}"
            )
        for key in ("emoji", "description"):
            if key not in h or not isinstance(h[key], str):
                raise RuntimeError(f"{path}: houses[{i}].{key} must be a string")

        house_prim[name] = scores_f
        order.append(name)

    return axes, house_prim, order, houses


PRIM_NAMES: list[str]
HOUSE_PRIM: dict[str, list[float]]
HOUSE_ORDER: list[str]
_HOUSE_ROWS: list[dict]

PRIM_NAMES, HOUSE_PRIM, HOUSE_ORDER, _HOUSE_ROWS = _load_houses(_HOUSES_JSON)
N_PRIM = len(PRIM_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
#  PCA via SVD (once at import)
# ══════════════════════════════════════════════════════════════════════════════
def run_pca(prim_dict: dict, house_order: list, var_threshold: float = 0.85) -> dict:
    M        = np.array([prim_dict[h] for h in house_order], dtype=float)
    col_mean = M.mean(axis=0)
    U, s, Vt = np.linalg.svd(M - col_mean, full_matrices=False)
    var      = (s**2) / (s**2).sum()
    cumvar   = np.cumsum(var)
    k        = min(int(np.searchsorted(cumvar, var_threshold)) + 1, len(s))
    return dict(k=k, s=s[:k], var=var[:k], cumvar=float(cumvar[k-1]),
                Vt=Vt[:k], col_mean=col_mean, H=(U[:, :k] * s[:k]))

def label_dims(Vt: np.ndarray, names: list, top_n: int = 2) -> list:
    return [" / ".join(f"{'+'if Vt[i,j]>0 else '−'}{names[j]}"
                       for j in np.argsort(np.abs(Vt[i]))[::-1][:top_n])
            for i in range(len(Vt))]

def project(raw: list, pca: dict) -> list:
    return ((np.array(raw) - pca["col_mean"]) @ pca["Vt"].T).tolist()

_PCA        = run_pca(HOUSE_PRIM, HOUSE_ORDER)
_DIM_LABELS = label_dims(_PCA["Vt"], PRIM_NAMES)
_K          = _PCA["k"]

DEFAULT_HOUSES = [
    {
        "name": h["name"],
        "emoji": h["emoji"],
        "color": TERMINAL_COLORS[h["terminal_color"].lower()],
        "desc": h["description"],
        "profile": _PCA["H"][i].tolist(),
    }
    for i, h in enumerate(_HOUSE_ROWS)
]
