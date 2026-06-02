"""
Lumerical RCWA for Sample A and B.
Key fixes applied:
  1. Geometry added BEFORE RCWA solver (official recommended workflow)
  2. Interface positions explicitly set via setnamed("RCWA","interface absolute positions",...)
  3. Removed invalid properties: "incident angle", "center wavelength"
"""
import sys, os, time
sys.path.insert(0, 'C:/Program Files/Lumerical/v241/api/python')
import lumapi, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Sample parameters
SAMPLES = {
    "A": {"period": 680.0e-9, "pillar": 309.1e-9, "height": 209.7e-9},
    "B": {"period": 675.0e-9, "pillar": 351.1e-9, "height": 282.6e-9},
}
SUB_THICK = 2e-6
BUF       = 1e-6

def run_rcwa_sample(sim, sp):
    period = sp["period"]
    pillar = sp["pillar"]
    height = sp["height"]
    zmin   = -SUB_THICK - BUF
    zmax   =  height + BUF

    lsf = f"""
switchtolayout;
selectall; delete;

# 1. Add geometry FIRST (official workflow: structures before RCWA solver)
addrect;
set("name","substrate");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{zmin}); set("z max",0);
set("material","SiO2 (Glass) - Palik");

addrect;
set("name","pillar");
set("x",0); set("x span",{pillar});
set("y",0); set("y span",{period*4});
set("z min",0); set("z max",{height});
set("material","SiO2 (Glass) - Palik");

# 2. Add RCWA solver AFTER geometry
addrcwa;
set("name","RCWA");
set("simulation region","3D");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{zmin}); set("z max",{zmax});
set("mesh cells x",100); set("mesh cells y",100); set("mesh cells z",100);
set("frequency points",256);
set("angle theta",0); set("angle phi",0);

# 3. Explicitly set layer interface positions
setnamed("RCWA","interface absolute positions",[0,{height}]);

run;
"""
    t0 = time.perf_counter()
    sim.eval(lsf)
    dt = time.perf_counter() - t0
    print(f"  run() in {dt:.2f}s")

    # Verify grating detection
    z_layers = np.array(sim.getresult("RCWA","z")).flatten() * 1e9
    grating_ok = any(abs(z_layers - height*1e9) < 30)
    print(f"  z layers (nm): {np.round(z_layers,1).tolist()}")
    print(f"  Grating detected: {grating_ok}")

    # Extract results
    te  = sim.getresult("RCWA","total_energy")
    lam = np.array(te['lambda']).flatten() * 1e9
    Rs  = np.array(te['Rs']).flatten()
    Rp  = np.array(te['Rp']).flatten()
    Ts  = np.array(te['Ts']).flatten()
    Tp  = np.array(te['Tp']).flatten()
    R   = (Rs + Rp) / 2

    sort_i = np.argsort(lam)
    lam, Rs, Rp, Ts, Tp, R = lam[sort_i], Rs[sort_i], Rp[sort_i], Ts[sort_i], Tp[sort_i], R[sort_i]

    print(f"  Rs range: [{Rs.min():.4f}, {Rs.max():.4f}]  constant={np.allclose(Rs,Rs[0],atol=1e-3)}")
    return {"lam": lam, "Rs": Rs, "Rp": Rp, "Ts": Ts, "Tp": Tp, "R": R}


# Run Sample A and B with a single session
sim = lumapi.FDTD(hide=True)
sim.setglobalsource("wavelength start", 400e-9)
sim.setglobalsource("wavelength stop",  750e-9)
sim.setglobalmonitor("use wavelength spacing", True)
sim.setglobalmonitor("use source limits", True)
sim.setglobalmonitor("frequency points", 256)

results = {}
for name, sp in SAMPLES.items():
    print(f"\nSample {name}: period={sp['period']*1e9:.0f}nm  "
          f"pillar={sp['pillar']*1e9:.1f}nm  height={sp['height']*1e9:.1f}nm")
    try:
        results[name] = run_rcwa_sample(sim, sp)
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = None

sim.close()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, sp) in zip(axes, SAMPLES.items()):
    res = results.get(name)
    if res is not None:
        ax.plot(res["lam"], res["Rs"], 'b--', lw=1.5, label='Lumerical RCWA (s-pol)')
        ax.plot(res["lam"], res["Rp"], 'r--', lw=1.5, label='Lumerical RCWA (p-pol)')
        ax.plot(res["lam"], res["R"],  'g-',  lw=2,   label='Lumerical RCWA (avg)')
    ax.set_title(f'Sample {name}: p={sp["period"]*1e9:.0f}nm  '
                 f'w={sp["pillar"]*1e9:.1f}nm  h={sp["height"]*1e9:.1f}nm')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.legend(fontsize=8)
    ax.set_xlim(400, 750)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r"C:\Users\user\OneDrive\Desktop\AOI_project\validation\rcwa_lumerical_AB.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()
