"""
3-way overlay: Measured (Mattila 2024) vs torcwa vs ANN
"""
import numpy as np, torch, torcwa, h5py, yaml, io, os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DB   = r"C:\Users\user\.refractiveindex.info-database\data"

# ── ANN ───────────────────────────────────────────────────────────────────────
from eval_ann_sio2 import ForwardANN, normalise
ckpt  = torch.load(os.path.join(BASE, "ann_sio2.pth"), map_location="cpu", weights_only=False)
model = ForwardANN(ckpt["n_wl"]); model.load_state_dict(ckpt["model"]); model.eval()
wl_ann = np.linspace(ckpt["wl_min"], ckpt["wl_max"], ckpt["n_wl"])

# ── SiO2 eps ──────────────────────────────────────────────────────────────────
WL_NM = np.linspace(400, 750, 256); WL_UM = WL_NM / 1000.0
with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
    data = yaml.safe_load(f)
for blk in data["DATA"]:
    rows = np.array([[float(v) for v in ln.split()]
                     for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
    wl_db, n_db, k_db = rows[:, 0]*1000, rows[:, 1], rows[:, 2]
eps_sio2 = (interp1d(wl_db, n_db, "cubic", fill_value="extrapolate")(WL_NM) +
            1j*interp1d(wl_db, k_db, "cubic", fill_value="extrapolate")(WL_NM))**2

NG, NX = 9, 21
dev, dtype = torch.device("cpu"), torch.complex64

def sim_torcwa(period_nm, pillar_nm, height_nm):
    p, w, h = period_nm/1e3, pillar_nm/1e3, height_nm/1e3
    si_cols = max(1, min(int(NX*w/p), NX-1))
    mask = torch.zeros(NX, NX, dtype=torch.bool); mask[:, :si_cols] = True
    R = []
    for i, wl in enumerate(WL_UM):
        ep = complex(eps_sio2[i]); ep_t = torch.tensor(ep, dtype=dtype)
        sim = torcwa.rcwa(freq=1.0/wl, order=[1, NG], L=[p, p], dtype=dtype, device=dev)
        sim.add_input_layer(eps=torch.tensor(1.0, dtype=dtype))
        sim.add_output_layer(eps=ep_t)
        sim.set_incident_angle(inc_ang=0.01, azi_ang=0.0)
        ep_g = torch.ones(NX, NX, dtype=dtype); ep_g[mask] = ep_t
        sim.add_layer(thickness=h, eps=ep_g)
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        sim.solve_global_smatrix()
        rv = sim.S_parameters(orders=[0,0], direction="forward",
                              port="reflection", polarization="xx", ref_order=[0,0])
        R.append(float(abs(rv)**2))
    return np.array(R)

def ann_predict(period_nm, pillar_nm, height_nm, wl_target):
    x = normalise(period_nm, pillar_nm, height_nm).unsqueeze(0)
    with torch.no_grad():
        R = model(x).squeeze(0).cpu().numpy()
    R = np.clip(savgol_filter(np.clip(R, 0, 1), 15, 3), 0, 1)
    return interp1d(wl_ann, R, kind="cubic", fill_value="extrapolate")(wl_target)

# ── Measured data from h5 ─────────────────────────────────────────────────────
samples = []
h5_A = os.path.join(BASE, "validation", "v1_A_256.h5")
if os.path.exists(h5_A):
    with h5py.File(h5_A, "r") as f:
        wl_m = f["lambda"][:].flatten()
        params = f["params"]["d680_el300"][0]; eta_r = f["eta_r"]["d680_el300"][0]
    period, height, groove = params[0], params[1], params[2]
    samples.append({"label": f"Sample A  p={period:.0f} ridge={period-groove:.0f} h={height:.0f}nm",
                    "period": period, "pillar": period-groove, "height": height,
                    "wl_m": wl_m, "eta_r": eta_r})

h5_B = os.path.join(BASE, "validation", "v1_B_256.h5")
if os.path.exists(h5_B):
    with h5py.File(h5_B, "r") as f:
        wl_m_b = f["lambda"][:].flatten()
        best_key, best_err = None, 1e9
        for k in f["eta_r"].keys():
            p = f["params"][k][0]; err = abs(p[0]-675)+abs((p[0]-p[2])-351)+abs(p[1]-283)
            if err < best_err: best_err, best_key = err, k
        params_b = f["params"][best_key][0]; eta_r_b = f["eta_r"][best_key][0]
    pb = params_b[0]
    samples.append({"label": f"Sample B  p={pb:.0f} ridge={pb-params_b[2]:.0f} h={params_b[1]:.0f}nm",
                    "period": pb, "pillar": pb-params_b[2], "height": params_b[1],
                    "wl_m": wl_m_b, "eta_r": eta_r_b})

# ── Plot ──────────────────────────────────────────────────────────────────────
print(f"Running {len(samples)} torcwa simulations...")
fig, axes = plt.subplots(1, len(samples), figsize=(7*len(samples), 5))
if len(samples) == 1: axes = [axes]

for ax, s in zip(axes, samples):
    print(f"  {s['label']}")
    R_t = sim_torcwa(s["period"], s["pillar"], s["height"])
    R_a = ann_predict(s["period"], s["pillar"], s["height"], s["wl_m"])
    mae_t = np.abs(R_t[np.searchsorted(WL_NM, s["wl_m"])] - s["eta_r"]).mean()*100
    mae_a = np.abs(R_a - s["eta_r"]).mean()*100

    ax.plot(s["wl_m"],  s["eta_r"]*100, "C1-",  lw=2,   label="Measured")
    ax.plot(WL_NM,      R_t*100,        "C0-",  lw=1.5, label=f"torcwa (MAE={mae_t:.2f}%)")
    ax.plot(s["wl_m"],  R_a*100,        "C2--", lw=1.5, label=f"ANN (MAE={mae_a:.2f}%)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance (%)")
    ax.set_title(s["label"]); ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
out = os.path.join(BASE, "compare_3way.png")
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
