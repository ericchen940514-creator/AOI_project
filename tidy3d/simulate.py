"""
Tidy3D FDTD simulation — single (w, h, p) case.

Set API key before running:
    $env:TIDY3D_API_KEY="your_api_key"   # PowerShell

Run:
    python tidy3d/simulate.py
    python tidy3d/simulate.py --w 0.15 --h 0.30 --p 0.40
    python tidy3d/simulate.py --reference-only   # submit reference sim only
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tidy3d as td
import tidy3d.web as web
from tidy3d import material_library

API_KEY_ENV = "TIDY3D_API_KEY"
OUTPUT_DIR  = "tidy3d/output"

NUM_FREQS = 100
FREQS = np.linspace(td.C_0 / 0.700, td.C_0 / 0.400, NUM_FREQS)  # 400–700 nm
WAVELENGTHS_NM = (td.C_0 / FREQS) * 1e3


def configure():
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV}.\n"
            f'Set it in PowerShell:  $env:{API_KEY_ENV}="your_api_key"'
        )
    web.configure(apikey=key)


def build_simulation(w: float, h: float, p: float,
                     include_substrate: bool, include_grating: bool) -> td.Simulation:
    silicon = material_library["cSi"]["Green2008"]

    substrate = td.Structure(
        geometry=td.Box(center=[0, 0, -5], size=[td.inf, td.inf, 10]),
        medium=silicon,
    )
    grating = td.Structure(
        geometry=td.Box(center=[0, 0, h / 2], size=[td.inf, w, h]),
        medium=silicon,
    )
    plane_wave = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=5e14, fwidth=2e14),
        size=(td.inf, td.inf, 0),
        center=[0, 0, h + 1.0],
        direction="-",
        pol_angle=np.pi / 2,
    )
    monitor_r = td.FluxMonitor(
        center=[0, 0, h + 1.5],
        size=[td.inf, td.inf, 0],
        freqs=FREQS,
        name="reflectance_monitor",
    )

    structures = []
    if include_substrate:
        structures.append(substrate)
    if include_grating:
        structures.append(grating)

    return td.Simulation(
        size=[p, p, h + 4.0],
        structures=structures,
        sources=[plane_wave],
        monitors=[monitor_r],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
    )


def run_job(sim: td.Simulation, task_name: str, hdf5_path: str,
            max_retries: int = 3) -> td.SimulationData:
    for attempt in range(1, max_retries + 1):
        try:
            job = web.Job(simulation=sim, task_name=task_name)
            print(f"  Submitting '{task_name}' (attempt {attempt})")
            t0 = time.perf_counter()
            data = job.run(path=hdf5_path)
            print(f"  Done in {time.perf_counter() - t0:.1f} s")
            return data
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(5)


def extract_reflectance(device_data: td.SimulationData,
                        reference_data: td.SimulationData) -> np.ndarray:
    ref_flux = np.asarray(reference_data["reflectance_monitor"].flux, dtype=np.float64)
    dev_flux = np.asarray(device_data["reflectance_monitor"].flux,    dtype=np.float64)
    reflected = dev_flux - ref_flux
    R = -reflected / np.maximum(np.abs(ref_flux), 1e-12)
    return np.clip(np.real(R), 0.0, 1.5)


def save_outputs(w: float, h: float, p: float, reflectance: np.ndarray) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"w{w*1000:.0f}_h{h*1000:.0f}_p{p*1000:.0f}"

    csv_path = f"{OUTPUT_DIR}/tidy3d_{tag}.csv"
    pd.DataFrame({"wavelength_nm": WAVELENGTHS_NM, "reflectance_fdtd": reflectance}).to_csv(
        csv_path, index=False
    )
    print(f"Saved: {csv_path}")

    png_path = f"{OUTPUT_DIR}/tidy3d_{tag}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTHS_NM, reflectance, linewidth=2, color="tab:blue", label="FDTD Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"Tidy3D\nw={w*1000:.0f}nm  h={h*1000:.0f}nm  p={p*1000:.0f}nm")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png_path}")


def main(w: float, h: float, p: float, reference_only: bool) -> None:
    configure()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"w{w*1000:.0f}_h{h*1000:.0f}_p{p*1000:.0f}"

    print(f"w={w} um  h={h} um  p={p} um")

    ref_sim  = build_simulation(w, h, p, include_substrate=False, include_grating=False)
    ref_data = run_job(ref_sim, f"ref_{tag}", f"{OUTPUT_DIR}/ref_{tag}.hdf5")

    if reference_only:
        print("--reference-only: stopping after reference simulation.")
        return

    dev_sim  = build_simulation(w, h, p, include_substrate=True, include_grating=True)
    dev_data = run_job(dev_sim, f"dev_{tag}", f"{OUTPUT_DIR}/dev_{tag}.hdf5")

    reflectance = extract_reflectance(dev_data, ref_data)
    save_outputs(w, h, p, reflectance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, default=0.150)
    parser.add_argument("--h", type=float, default=0.300)
    parser.add_argument("--p", type=float, default=0.400)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    main(args.w, args.h, args.p, args.reference_only)
