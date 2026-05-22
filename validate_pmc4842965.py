"""
Validation against PMC4842965 (Brissinger et al., Sci. Rep. 6:24947, 2016).
"Optimized 2D array of thin silicon pillars for efficient antireflective coatings"

Structure : Si 2D square pillar array on Si substrate, square lattice
Geometry  : w=140nm, h=110nm, p=240nm  (optimized design from paper)
Expected  : avg R ~ 4% in 400-900nm (paper claim, Fig 3a s-pol normal incidence)

── How to get precise reference data for overlay ─────────────────────────────
1. Open paper at https://pmc.ncbi.nlm.nih.gov/articles/PMC4842965/
2. Find Figure 3a  (reflectance vs wavelength, s-pol, normal incidence)
3. Open https://automeris.io/WebPlotDigitizer/ → digitize the "optimized" curve
4. Export CSV → save here as  ref_pmc4842965_fig3a.csv
   Columns: wavelength_nm, reflectance   (reflectance as 0–1, NOT percent)
5. Re-run → comparison overlay will use your precise data
──────────────────────────────────────────────────────────────────────────────
"""

import io, time
import os
import numpy as np
import yaml
import pandas as pd
import S4
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Geometry (PMC4842965 optimized design) ─────────────────────────────────────
W, H, P = 0.140, 0.110, 0.240      # µm: w=140 h=110 p=240 nm

# ── Spectral range ─────────────────────────────────────────────────────────────
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 201
wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)

# ── S4 Fourier orders ──────────────────────────────────────────────────────────
NUM_BASIS = 201

# ── n,k data (Aspnes Si, same database as generate.py) ────────────────────────
DB = "/mnt/c/Users/user/.refractiveindex.info-database/data"

def load_si_nk():
    path = f"{DB}/main/Si/nk/Aspnes.yml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        if blk["type"].strip() == "tabulated nk":
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
            wl, n, k = rows[:, 0], rows[:, 1], rows[:, 2]
            return (interp1d(wl, n, kind="cubic", fill_value="extrapolate"),
                    interp1d(wl, k, kind="cubic", fill_value="extrapolate"))
    raise RuntimeError("Aspnes nk not found")

n_fn, k_fn = load_si_nk()
eps_si = (n_fn(wavelengths) + 1j * k_fn(wavelengths)) ** 2


# ── Sanity check: Fresnel R at flat Si surface ─────────────────────────────────
def fresnel_R(ep):
    n = np.sqrt(ep)
    return abs((1 - n) / (1 + n)) ** 2

R_flat = np.array([fresnel_R(complex(eps_si[i])) for i in range(NUM_WL)])
print(f"Flat Si avg R (400-700nm): {R_flat.mean()*100:.1f}%  (expect ~35%)")

# ── Reference data: approx from paper Fig 3a + optional precise CSV ───────────
# Values read off Fig 3a (s-pol, normal incidence, optimised structure)
_ref_wl = np.array([400, 450, 500, 550, 600, 650, 700])
_ref_R  = np.array([0.060, 0.035, 0.025, 0.030, 0.040, 0.050, 0.055])
ref_fn  = interp1d(_ref_wl / 1000, _ref_R, kind="cubic",
                   bounds_error=False, fill_value="extrapolate")
REF_LABEL = "Paper Fig3a (approx — digitize for precision)"

REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ref_pmc4842965_fig3a.csv")
if os.path.exists(REF_CSV):
    df_ref  = pd.read_csv(REF_CSV)
    ref_fn  = interp1d(df_ref["wavelength_nm"] / 1000, df_ref["reflectance"],
                       kind="cubic", bounds_error=False, fill_value="extrapolate")
    REF_LABEL = "Paper Fig3a (digitized)"
    print(f"Loaded digitized reference: {REF_CSV}")
else:
    print("No ref_pmc4842965_fig3a.csv found — using approximate reference values")


# ── S4 2D square pillar ────────────────────────────────────────────────────────
def run_s4_2d_pillar(w, h, p, num_basis):
    """
    Square Si pillar (w x w) centered in a square unit cell (p x p).
    Si substrate below, air above.  s-pol normal incidence.
    At normal incidence a 2D square-symmetric structure gives identical
    s- and p-pol results, so s-pol is sufficient.
    """
    results = []
    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        s = S4.New(Lattice=((p, 0), (0, p)), NumBasis=num_basis)
        s.SetMaterial(Name='air', Epsilon=1.0 + 0j)
        s.SetMaterial(Name='si',  Epsilon=ep)

        s.AddLayer(Name='above',     Thickness=0, Material='air')
        s.AddLayer(Name='grating',   Thickness=h, Material='air')
        s.AddLayer(Name='substrate', Thickness=0, Material='si')

        # Square pillar centered at (p/2, p/2), side length w
        s.SetRegionRectangle(
            Layer='grating',
            Material='si',
            Center=(p / 2, p / 2),
            Angle=0,
            Halfwidths=(w / 2, w / 2),
        )

        s.SetExcitationPlanewave(
            IncidenceAngles=(0, 0),
            sAmplitude=1,
            pAmplitude=0,
            Order=0,
        )
        s.SetFrequency(1.0 / wl)
        fwd, bwd = s.GetPowerFlux(Layer='above', zOffset=0)
        results.append(float(-np.real(bwd)))

    return np.clip(np.array(results), 0, 1)


# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\nRunning S4 — w={W*1e3:.0f}nm  h={H*1e3:.0f}nm  p={P*1e3:.0f}nm"
      f"  NumBasis={NUM_BASIS}  {NUM_WL} wavelengths")
t0 = time.perf_counter()
R_pillar = run_s4_2d_pillar(W, H, P, NUM_BASIS)
dt = time.perf_counter() - t0

R_ref = ref_fn(wavelengths)
mae   = np.abs(R_pillar - R_ref).mean() * 100
avg_R = R_pillar.mean() * 100

print(f"Done in {dt:.1f}s")
print(f"S4 avg R  (400-700nm, s-pol) : {avg_R:.2f}%")
print(f"MAE vs reference             : {mae:.2f}%")
print(f"Flat Si avg R                : {R_flat.mean()*100:.1f}%")
print(f"Paper claim                  : avg R ~ 4% in 400-900nm")
print(f"=> {'PASS' if avg_R < 8 else 'CHECK'}")


# ── Plot ───────────────────────────────────────────────────────────────────────
wl_nm = wavelengths * 1000
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(wl_nm, R_flat  * 100, 'k--', lw=1.2, label='Flat Si (Fresnel)')
ax.plot(wl_nm, R_ref   * 100, 'C1-', lw=2.5, label=REF_LABEL)
ax.plot(wl_nm, R_pillar* 100, 'C0--',lw=2.0,
        label=f'S4 (this work)  avg={avg_R:.1f}%  MAE={mae:.1f}%')

ax.set_xlabel("Wavelength (nm)", fontsize=13)
ax.set_ylabel("Reflectance (%)", fontsize=13)
ax.set_title(
    "PMC4842965 — Si 2D square pillar  w=140/h=110/p=240nm  s-pol  NumBasis=201\n"
    "Orange = paper, Blue dashed = S4",
    fontsize=11,
)
ax.set_ylim(0, 42)
ax.legend(fontsize=11)
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()

out = "/mnt/c/Users/user/OneDrive/Desktop/AOI_project/validate_pmc4842965.png"
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
