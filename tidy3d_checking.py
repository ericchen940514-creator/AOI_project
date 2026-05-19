import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
from tidy3d import material_library


# Single-run configuration. Units: um.
W = 0.150
H = 0.300
P = 0.400

# If True, submits only the reference simulation and exits.
REFERENCE_ONLY = False

# Tidy3D API key should be set in the environment before running.
# PowerShell example:
#   $env:TIDY3D_API_KEY="your_api_key"
API_KEY_ENV = "TIDY3D_API_KEY"
os.environ.setdefault(API_KEY_ENV, "CO6qae1AvXVW72e8cGJaUoiGrL99QhMNazK8KfWttklZX9St")

NUM_FREQS = 100
# td.C_0 is in um/s; wavelengths must be in um to get Hz
FREQS = np.linspace(td.C_0 / 0.700, td.C_0 / 0.400, NUM_FREQS)  # 400-700 nm
WAVELENGTHS_NM = (td.C_0 / FREQS) * 1e3  # um -> nm


def configure_tidy3d():
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV}. Set it in your environment before running this script."
        )
    web.configure(apikey=api_key)


def build_common_objects(height_um):
    silicon = material_library["cSi"]["Green2008"]

    substrate = td.Structure(
        geometry=td.Box(center=[0, 0, -5], size=[td.inf, td.inf, 10]),
        medium=silicon,
    )

    grating = td.Structure(
        geometry=td.Box(center=[0, 0, height_um / 2], size=[td.inf, W, H]),
        medium=silicon,
    )

    plane_wave = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=5e14, fwidth=2e14),
        size=(td.inf, td.inf, 0),
        center=[0, 0, height_um + 1.0],
        direction="-",
        pol_angle=np.pi / 2,  # s-polarization (E along y), matches RCWA s_amp=1
    )

    monitor_r = td.FluxMonitor(
        center=[0, 0, height_um + 1.5],
        size=[td.inf, td.inf, 0],
        freqs=FREQS,
        name="reflectance_monitor",
    )

    return substrate, grating, plane_wave, monitor_r


def build_simulation(include_substrate, include_grating):
    substrate, grating, plane_wave, monitor_r = build_common_objects(H)
    structures = []
    if include_substrate:
        structures.append(substrate)
    if include_grating:
        structures.append(grating)

    return td.Simulation(
        size=[P, P, H + 4.0],
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


def run_simulation(simulation, task_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            job = web.Job(simulation=simulation, task_name=task_name)
            print(f"Submitting task: {task_name} (attempt {attempt})")
            start_time = time.perf_counter()
            data = job.run(path=f"{task_name}.hdf5")
            elapsed_s = time.perf_counter() - start_time
            print(f"Finished task: {task_name} in {elapsed_s:.1f} s")
            return data
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(5)


def extract_flux(sim_data):
    flux = sim_data["reflectance_monitor"].flux
    return np.asarray(flux, dtype=np.float64)


def compute_reflectance(device_flux, reference_flux):
    eps = 1e-12
    incident_flux = np.asarray(reference_flux, dtype=np.float64)
    total_flux = np.asarray(device_flux, dtype=np.float64)
    reflected_flux = total_flux - incident_flux
    reflectance = -reflected_flux / np.maximum(np.abs(incident_flux), eps)
    return np.clip(np.real(reflectance), 0.0, 1.5)


def save_outputs(wavelengths_nm, reflectance):
    output_df = pd.DataFrame(
        {
            "wavelength_nm": wavelengths_nm,
            "reflectance_fdtd": reflectance,
        }
    )
    output_df.to_csv("tidy3d_single_result.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths_nm, reflectance, linewidth=2, color="tab:blue", label="FDTD Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"Tidy3D Single Run\nw={W*1000:.0f}nm, h={H*1000:.0f}nm, p={P*1000:.0f}nm")
    plt.tight_layout()
    plt.savefig("tidy3d_single_result.png", dpi=300, bbox_inches="tight")
    print("Saved tidy3d_single_result.csv")
    print("Saved tidy3d_single_result.png")


def main():
    configure_tidy3d()

    print(f"Running single Tidy3D case: w={W} um, h={H} um, p={P} um")
    reference_sim = build_simulation(include_substrate=False, include_grating=False)
    reference_data = run_simulation(reference_sim, "tidy3d_reference_single")

    if REFERENCE_ONLY:
        print("REFERENCE_ONLY=True, skipping device simulation.")
        return

    device_sim = build_simulation(include_substrate=True, include_grating=True)
    device_data = run_simulation(device_sim, "tidy3d_device_single")

    reference_flux = extract_flux(reference_data)
    device_flux = extract_flux(device_data)
    reflectance = compute_reflectance(device_flux, reference_flux)

    save_outputs(WAVELENGTHS_NM, reflectance)


if __name__ == "__main__":
    main()
