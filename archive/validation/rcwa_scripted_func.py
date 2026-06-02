"""
Lumerical scripted rcwa(geom, src) function.
This is the ANALYTICAL RCWA path - doesn't need GUI geometry detection.
Requires FDTD setup as API context.
Extract reflectance from rawdata returned by rcwa().
"""
import sys, time
sys.path.insert(0, 'C:/Program Files/Lumerical/v241/api/python')
import lumapi, numpy as np

c0 = 3e8
period = 680e-9; pillar = 309.1e-9; height = 209.7e-9
sub_thick = 2e-6; Y_SPAN = 20e-9

fdtd = lumapi.FDTD(hide=True)
print("Connected.")

# FDTD needed as API context
fdtd.addfdtd()
fdtd.set("x", 0.0); fdtd.set("x span", period)
fdtd.set("y", 0.0); fdtd.set("y span", Y_SPAN)
fdtd.set("z min", -sub_thick); fdtd.set("z max", height + 1e-6)
fdtd.set("x min bc", "Periodic"); fdtd.set("x max bc", "Periodic")
fdtd.set("y min bc", "Periodic"); fdtd.set("y max bc", "Periodic")

# Substrate (SiO2) below z=0
fdtd.addrect()
fdtd.set("name", "substrate")
fdtd.set("x", 0.0); fdtd.set("x span", period * 2)
fdtd.set("y", 0.0); fdtd.set("y span", Y_SPAN * 2)
fdtd.set("z min", -sub_thick * 1.5); fdtd.set("z max", 0.0)
fdtd.set("material", "SiO2 (Glass) - Palik")

# Pillar (SiO2) above z=0
fdtd.addrect()
fdtd.set("name", "pillar")
fdtd.set("x", 0.0); fdtd.set("x span", pillar)
fdtd.set("y", 0.0); fdtd.set("y span", Y_SPAN * 2)
fdtd.set("z min", 0.0); fdtd.set("z max", height)
fdtd.set("material", "SiO2 (Glass) - Palik")
print("FDTD + geometry set up.")

# Geometry struct for rcwa(): defines RCWA domain + layer boundaries
geom = {
    "injection_axis": "z-axis",
    "x_min": np.float64(-period / 2),
    "x_max": np.float64(period / 2),
    "y_min": np.float64(-period / 2),
    "y_max": np.float64(period / 2),
    "z_min": np.float64(-sub_thick),
    "z_max": np.float64(height + 1e-6),
    "layer_positions": np.array([0.0, height]),  # layer interfaces
}
WL = 500e-9
src_single = {
    "f": np.float64(c0 / WL),
    "theta": np.float64(0.0),
    "phi": np.float64(0.0),
}

print(f"\n=== rcwa() at {WL*1e9:.0f}nm ===")
rawdata = None
try:
    t0 = time.perf_counter()
    rawdata = fdtd.rcwa(geom, src_single)
    dt = time.perf_counter() - t0
    print(f"  SUCCESS in {dt:.3f}s")
    print(f"  type = {type(rawdata).__name__}")
    if isinstance(rawdata, dict):
        print(f"  keys = {list(rawdata.keys())}")
        for k, v in rawdata.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                if v.size <= 20:
                    print(f"      = {v.flatten()}")
                else:
                    print(f"      first5={v.flatten()[:5]}")
            elif isinstance(v, dict):
                print(f"    {k}: dict keys={list(v.keys())}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:60]!r}")
except Exception as e:
    print(f"  FAILED: {e}")

# If rawdata is available, try rcwasweeppropagation
if rawdata is not None:
    print("\n=== rcwasweeppropagation ===")
    # Try various N arguments
    for N_try in [
        np.int32(1),
        np.array([1]),
        np.array([[1]]),
        np.array([0]),
        np.float64(1.0),
        np.array([height + 0.5e-6]),   # monitor position z
        np.array([[0.0, height]]),      # layer positions
        1,
    ]:
        try:
            r = fdtd.rcwasweeppropagation(rawdata, N_try)
            print(f"  N={type(N_try).__name__}({N_try!r if np.size(N_try) <= 3 else '...'}): SUCCESS")
            if isinstance(r, dict):
                print(f"    keys={list(r.keys())}")
            elif hasattr(r, 'shape'):
                print(f"    shape={r.shape}, first5={r.flatten()[:5]}")
            break
        except Exception as e:
            print(f"  N={type(N_try).__name__}: {str(e)[:80]}")

# Multi-wavelength rcwa()
print("\n=== rcwa() sweep 400-750nm ===")
WLS = np.linspace(400e-9, 750e-9, 50)
Rs_all = []
for wl in WLS[:5]:  # test 5 wavelengths first
    src_wl = {"f": np.float64(c0/wl), "theta": np.float64(0.0), "phi": np.float64(0.0)}
    try:
        rd = fdtd.rcwa(geom, src_wl)
        if isinstance(rd, dict) and 'data' in rd:
            Rs_all.append(float(wl))
            print(f"  {wl*1e9:.0f}nm: OK, keys={list(rd.keys())}")
        else:
            print(f"  {wl*1e9:.0f}nm: returned {type(rd).__name__}")
    except Exception as e:
        print(f"  {wl*1e9:.0f}nm: {str(e)[:60]}")
        break

fdtd.close()
print("\nDone.")
