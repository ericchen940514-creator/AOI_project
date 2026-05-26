"""
S4 vs tidy3d: accuracy and speed benchmark.
W=150nm, H=300nm, P=400nm, 100 wavelengths 400-700nm, Si grating, s-polarization.
"""
import io, time
import numpy as np
import pandas as pd
import yaml
import S4
from scipy.interpolate import interp1d

W, H, P = 0.150, 0.300, 0.400
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 100
wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)

TIDY3D_CSV = "/mnt/c/Users/user/OneDrive/Desktop/AOI_project/tidy3d/output/tidy3d_w150_h300_p400.csv"
DB = "/mnt/c/Users/user/.refractiveindex.info-database/data"  # WSL2 path


NK_FILE = "Aspnes"  # or "Green-2008"

def load_si_nk():
    with open(f"{DB}/main/Si/nk/{NK_FILE}.yml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        if blk["type"].strip() == "tabulated nk":
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
            wl, n, k = rows[:, 0], rows[:, 1], rows[:, 2]
            return (interp1d(wl, n, kind="cubic", fill_value="extrapolate"),
                    interp1d(wl, k, kind="cubic", fill_value="extrapolate"))
    raise RuntimeError(f"{NK_FILE} nk not found")


n_fn, k_fn = load_si_nk()
eps_si = (n_fn(wavelengths) + 1j * k_fn(wavelengths)) ** 2


# ── Fresnel verification with flat Si slab ─────────────────────────────────────
def fresnel_R(ep):
    n = np.sqrt(ep)
    return abs((1 - n) / (1 + n)) ** 2


def s4_flat_R(wl, ep):
    """Flat Si slab (no grating) — verify flux normalization."""
    s = S4.New(Lattice=((P, 0), (0, P)), NumBasis=1)
    s.SetMaterial(Name='air', Epsilon=1.0 + 0j)
    s.SetMaterial(Name='si',  Epsilon=ep)
    s.AddLayer(Name='above',     Thickness=0, Material='air')
    s.AddLayer(Name='slab',      Thickness=H, Material='si')
    s.AddLayer(Name='substrate', Thickness=0, Material='si')
    s.SetExcitationPlanewave(IncidenceAngles=(0, 0), sAmplitude=1, pAmplitude=0, Order=0)
    s.SetFrequency(1.0 / wl)
    fwd, bwd = s.GetPowerFlux(Layer='above', zOffset=0)
    return fwd, bwd


wl_test = 0.55
ep_test = complex(eps_si[np.argmin(np.abs(wavelengths - wl_test))])
fwd, bwd = s4_flat_R(wl_test, ep_test)
R_fresnel = fresnel_R(ep_test)
print(f"── Fresnel verification at {wl_test*1e3:.0f}nm ──")
print(f"  S4 fwd={fwd:.4f}  bwd={bwd:.4f}")
print(f"  Expected Fresnel R = {R_fresnel:.4f}")
print(f"  => Use R = -bwd.real = {-np.real(bwd):.4f}")
print()


def run_s4(w, h, p, num_basis):
    """
    1D Si grating on Si substrate.
    Ridge runs along y (infinite), periodic in x with period p, width w.
    s-pol: E along y (parallel to ridge).
    Matches tidy3d: size=[inf, w, h] grating, periodic x and y.
    """
    results = []
    fill = w / p

    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        s = S4.New(Lattice=((p, 0), (0, p)), NumBasis=num_basis)
        s.SetMaterial(Name='air', Epsilon=1.0 + 0j)
        s.SetMaterial(Name='si',  Epsilon=ep)

        s.AddLayer(Name='above',     Thickness=0, Material='air')
        s.AddLayer(Name='grating',   Thickness=h, Material='air')
        s.AddLayer(Name='substrate', Thickness=0, Material='si')

        # Ridge: width w in y, fully spanning x (1D grating periodic in y)
        # Center at (p/2, w/2), halfwidths (p/2, w/2)
        s.SetRegionRectangle(
            Layer='grating',
            Material='si',
            Center=(p / 2, w / 2),
            Angle=0,
            Halfwidths=(p / 2, w / 2),
        )

        # s-pol at normal incidence: E along y (parallel to ridge)
        s.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),
            sAmplitude=1,
            pAmplitude=0,
            Order=0,
        )
        s.SetFrequency(1.0 / wl)

        fwd, bwd = s.GetPowerFlux(Layer='above', zOffset=0)
        results.append(float(-np.real(bwd)))   # reflected backward flux

    return np.clip(np.array(results), 0, 1)


# ── Load tidy3d reference ──────────────────────────────────────────────────────
df_t = pd.read_csv(TIDY3D_CSV)
wl_t = df_t["wavelength_nm"].values / 1000.0
R_ref = interp1d(wl_t, df_t["reflectance_fdtd"].values,
                 kind="cubic", fill_value="extrapolate")(wavelengths)

print("=" * 60)
print(f"W={W*1e3:.0f}nm  H={H*1e3:.0f}nm  P={P*1e3:.0f}nm  {NUM_WL} wavelengths")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = {}
for nb in [21, 51, 101, 201, 301]:
    t0 = time.perf_counter()
    R = run_s4(W, H, P, nb)
    dt = time.perf_counter() - t0
    mae = np.abs(R - R_ref).mean() * 100
    results[nb] = R
    print(f"S4  NumBasis={nb:3d}: MAE={mae:.2f}%  time={dt:.1f}s", flush=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
wl_nm = wavelengths * 1000
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(wl_nm, R_ref, "k-", linewidth=2.5, label="Tidy3D FDTD (reference)")
styles = {21: ("C0", "--"), 51: ("C1", "-."), 101: ("C2", ":"), 201: ("C3", "-"), 301: ("C4", (0,(3,1,1,1)))}
for nb, R in results.items():
    mae = np.abs(R - R_ref).mean() * 100
    color, ls = styles[nb]
    ax.plot(wl_nm, R, color=color, linestyle=ls, linewidth=1.8,
            label=f"S4 NumBasis={nb}  MAE={mae:.2f}%")
ax.set_xlabel("Wavelength (nm)", fontsize=13)
ax.set_ylabel("Reflectance", fontsize=13)
ax.set_title(f"S4 vs Tidy3D  —  W={W*1e3:.0f}nm  H={H*1e3:.0f}nm  P={P*1e3:.0f}nm  s-pol", fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
out = "/mnt/c/Users/user/OneDrive/Desktop/AOI_project/s4_vs_tidy3d.png"
plt.savefig(out, dpi=150)
print(f"\nSaved plot: {out}", flush=True)
