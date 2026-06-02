"""
Single SiO2 grating FDTD simulation using Lumerical v241.
Mirrors the torcwa structure in sio2/generate_torcwa_sio2.py.

Structure:
  - Air above (n=1)
  - SiO2 grating pillar (pillar_nm wide, height_nm tall, period_nm pitch)
  - SiO2 substrate below
  - Normal incidence, TM polarisation (Ex), wavelength 400-750 nm

Usage:
    python lumerical/fdtd_single_sio2.py
    python lumerical/fdtd_single_sio2.py --period 680 --pillar 300 --height 200
"""
import sys, os, argparse
import numpy as np

LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.append(LUMAPI_PATH)
import lumapi

# ── Constants ─────────────────────────────────────────────────────────────────
WL_START_NM = 400.0
WL_STOP_NM  = 750.0
NUM_WL      = 100

# Simulation geometry margins (metres)
Z_ABOVE   = 1000e-9   # air region above grating
Z_BELOW   = 1000e-9   # substrate region below grating
Y_SPAN    = 20e-9     # collapsed y-dimension (1D grating)
SRC_ABOVE = 600e-9    # source z-offset above grating top
MON_ABOVE = 750e-9    # R monitor z-offset above grating top (above source)
MON_BELOW = -500e-9   # T monitor z-position (inside substrate)

MATERIAL_SIO2 = "SiO2 (Glass) - Palik"


# ── Build and run one FDTD simulation ────────────────────────────────────────
def simulate(period_nm: float, pillar_nm: float, height_nm: float,
             save_fsp: str = None) -> dict:
    """
    Returns dict with keys:
        wavelengths_nm : np.ndarray (NUM_WL,)
        R              : np.ndarray (NUM_WL,)  reflectance
        T              : np.ndarray (NUM_WL,)  transmittance
    """
    period = period_nm * 1e-9
    pillar = pillar_nm * 1e-9
    height = height_nm * 1e-9

    fdtd = lumapi.FDTD(hide=True)

    # ── FDTD region ───────────────────────────────────────────────────────────
    fdtd.addfdtd()
    fdtd.set("x", 0.0)
    fdtd.set("x span", period)
    fdtd.set("y", 0.0)
    fdtd.set("y span", Y_SPAN)
    fdtd.set("z min", -Z_BELOW)
    fdtd.set("z max", height + Z_ABOVE)
    fdtd.set("x min bc", "Periodic")
    fdtd.set("x max bc", "Periodic")
    fdtd.set("y min bc", "Periodic")
    fdtd.set("y max bc", "Periodic")
    # z boundaries default to PML

    # ── Materials ─────────────────────────────────────────────────────────────
    # SiO2 substrate (fills from below FDTD boundary to z=0)
    fdtd.addrect()
    fdtd.set("name", "substrate")
    fdtd.set("x span", period * 2)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z min", -Z_BELOW * 1.5)
    fdtd.set("z max", 0.0)
    fdtd.set("material", MATERIAL_SIO2)

    # SiO2 grating pillar (centred at x=0)
    fdtd.addrect()
    fdtd.set("name", "pillar")
    fdtd.set("x", 0.0)
    fdtd.set("x span", pillar)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z min", 0.0)
    fdtd.set("z max", height)
    fdtd.set("material", MATERIAL_SIO2)

    # ── Plane-wave source (TM, normal incidence, injecting downward) ──────────
    fdtd.addplane()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "z-axis")
    fdtd.set("direction", "Backward")           # -z = downward toward grating
    fdtd.set("x span", period * 2)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z", height + SRC_ABOVE)
    fdtd.set("polarization angle", 90.0)        # Ey = TE (s-pol), matches torcwa amplitude=[1,0] for grating periodic in y
    fdtd.set("wavelength start", WL_START_NM * 1e-9)
    fdtd.set("wavelength stop", WL_STOP_NM * 1e-9)

    # ── Reflection monitor (above source, captures upward reflected power) ────
    fdtd.addpower()
    fdtd.set("name", "R_mon")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", period * 2)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z", height + MON_ABOVE)
    fdtd.set("override global monitor settings", 1)
    fdtd.set("frequency points", NUM_WL)

    # ── Transmission monitor (inside substrate) ───────────────────────────────
    fdtd.addpower()
    fdtd.set("name", "T_mon")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", period * 2)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z", MON_BELOW)
    fdtd.set("override global monitor settings", 1)
    fdtd.set("frequency points", NUM_WL)

    # ── Run ───────────────────────────────────────────────────────────────────
    if save_fsp:
        fdtd.save(save_fsp)
    fdtd.run()

    # ── Extract results ───────────────────────────────────────────────────────
    # transmission() returns net power / source power
    # R_mon is above source → reflected light travels +z (opposite to source -z)
    # → sign is negative from Lumerical convention → R = -transmission
    T_R = fdtd.transmission("R_mon")
    T_T = fdtd.transmission("T_mon")

    # getdata returns frequency array; convert to wavelength in nm
    f_data  = np.array(fdtd.getdata("R_mon", "f")).flatten()
    wl_data = (3e8 / f_data) * 1e9       # m → nm, descending order
    # sort ascending by wavelength
    sort_idx = np.argsort(wl_data)
    wl_data  = wl_data[sort_idx]

    # R_mon above source: reflected light goes +z (opposite to Backward source) → positive
    # T_mon below substrate: transmitted light goes -z (same as source) → negative → negate
    R_vals =  np.array(T_R).flatten()[sort_idx]
    T_vals = -np.array(T_T).flatten()[sort_idx]

    fdtd.close()

    return {"wavelengths_nm": wl_data, "R": R_vals, "T": T_vals}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=680.0, help="Period (nm)")
    parser.add_argument("--pillar", type=float, default=300.0, help="Pillar width (nm)")
    parser.add_argument("--height", type=float, default=200.0, help="Grating height (nm)")
    parser.add_argument("--save",   default=None,              help="Save .fsp project file")
    args = parser.parse_args()

    print(f"Running FDTD: period={args.period}nm  pillar={args.pillar}nm  height={args.height}nm")
    result = simulate(args.period, args.pillar, args.height, save_fsp=args.save)

    wl = result["wavelengths_nm"]
    R  = result["R"]
    T  = result["T"]

    print(f"\nWavelength range: {wl.min():.1f} – {wl.max():.1f} nm  ({len(wl)} pts)")
    print(f"R range: {R.min():.4f} – {R.max():.4f}")
    print(f"T range: {T.min():.4f} – {T.max():.4f}")
    print(f"R+T check (should ~1): {(R+T).mean():.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(wl, R, label="R (FDTD)")
        plt.plot(wl, T, label="T (FDTD)")
        plt.plot(wl, R + T, "k--", lw=0.8, label="R+T")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power fraction")
        plt.title(f"SiO2 grating  p={args.period}  w={args.pillar}  h={args.height} nm")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "fdtd_single_result.png")
        plt.savefig(out, dpi=150)
        print(f"Saved plot → {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
