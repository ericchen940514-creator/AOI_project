"""
Material library — reads n/k data directly from the refractiveindex.info
YAML database that was downloaded by the `refractiveindex` PyPI package.

Install once:  pip install refractiveindex pyyaml
(The pip install triggers the database download; we then read it ourselves.)

Usage:
    from data_generation.materials import get_epsilon, MATERIALS
    eps = get_epsilon("Si", wavelength_um=0.5)

To add a new material:
  1. Go to https://refractiveindex.info and find your material.
  2. Note the URL path, e.g.  /main/Si/nk/Aspnes
  3. Add an entry to MATERIALS below: "MyMat": ("main", "Book", "subfolder/Page")
"""

from __future__ import annotations

import io
import os

import numpy as np
import yaml
from scipy.interpolate import interp1d

# ── Database root (downloaded by the refractiveindex package) ─────────────────
_DB_ROOT = os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data")


# ── Material catalogue ────────────────────────────────────────────────────────
# Key   : short name used in the rest of the pipeline
# Value : (shelf, book, subfolder/page)  — mirrors the URL path on the website
MATERIALS: dict[str, tuple[str, str, str]] = {
    "Si":    ("main", "Si",    "nk/Aspnes"),
    "SiO2":  ("main", "SiO2",  "nk/Franta"),
    "Si3N4": ("main", "Si3N4", "nk/Beliaev"),
    "TiO2":  ("main", "TiO2",  "nk/Franta"),
    "GaN":   ("main", "GaN",   "nk/Kawashima"),
    "Ge":    ("main", "Ge",    "nk/Aspnes"),
}

# ── Internal interpolation cache ──────────────────────────────────────────────
_cache: dict[str, tuple] = {}   # name → (n_interp, k_interp)


def _yml_path(name: str) -> str:
    if name not in MATERIALS:
        raise ValueError(f"Unknown material '{name}'. Available: {list(MATERIALS.keys())}")
    shelf, book, page = MATERIALS[name]
    return os.path.join(_DB_ROOT, shelf, book, page + ".yml")


def _load(name: str) -> tuple:
    if name in _cache:
        return _cache[name]

    path = _yml_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YAML not found: {path}\n"
            "Make sure you have run:  pip install refractiveindex\n"
            "(It downloads the database on first import.)"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl_n, n_arr, k_arr = None, None, None

    for block in data["DATA"]:
        dtype = block["type"].strip()
        rows = np.array(
            [[float(v) for v in line.split()]
             for line in io.StringIO(block["data"]).readlines() if line.strip()],
        )

        if dtype == "tabulated nk":
            wl_n, n_arr, k_arr = rows[:, 0], rows[:, 1], rows[:, 2]
        elif dtype == "tabulated n":
            wl_n, n_arr = rows[:, 0], rows[:, 1]
        elif dtype == "tabulated k":
            k_arr = rows[:, 1]

    if wl_n is None or n_arr is None:
        raise ValueError(f"Could not parse n data from {path}")

    n_fn = interp1d(wl_n, n_arr, kind="cubic", fill_value="extrapolate")
    k_fn = (interp1d(wl_n, k_arr, kind="cubic", fill_value="extrapolate")
            if k_arr is not None else lambda wl: 0.0)

    _cache[name] = (n_fn, k_fn)
    return _cache[name]


def get_nk(name: str, wavelength_um: float) -> tuple[float, float]:
    """Return (n, k) for *name* at *wavelength_um* micrometres."""
    n_fn, k_fn = _load(name)
    return float(n_fn(wavelength_um)), float(k_fn(wavelength_um))


def get_epsilon(name: str, wavelength_um: float) -> complex:
    """Return complex permittivity ε = (n + ik)² for *name* at *wavelength_um*."""
    n, k = get_nk(name, wavelength_um)
    return (n + 1j * k) ** 2


def list_materials() -> None:
    """Print all available materials."""
    print(f"{'Name':<10} {'Shelf':<8} {'Book':<10} {'Page'}")
    print("-" * 48)
    for name, (shelf, book, page) in MATERIALS.items():
        print(f"{name:<10} {shelf:<8} {book:<10} {page}")


if __name__ == "__main__":
    wl = 0.5
    list_materials()
    print(f"\nn/k at {wl} um:")
    for name in MATERIALS:
        try:
            n, k = get_nk(name, wl)
            print(f"  {name:<10} n={n:.4f}  k={k:.4f}")
        except Exception as e:
            print(f"  {name:<10} ERROR: {e}")
