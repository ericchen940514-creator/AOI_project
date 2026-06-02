"""
KEY FIX: set background material to '<Object defined dielectric>' on RCWA.
This makes RCWA compute structure from all geometry objects within bounds.
"""
import sys, time
sys.path.insert(0, 'C:/Program Files/Lumerical/v241/api/python')
import lumapi, numpy as np

period = 680e-9; pillar = 309.1e-9; height = 209.7e-9
sub_thick = 2e-6

fdtd = lumapi.FDTD(hide=True)
print("Connected.")

fdtd.setglobalsource("wavelength start", 400e-9)
fdtd.setglobalsource("wavelength stop",  750e-9)
fdtd.setglobalmonitor("use wavelength spacing", True)
fdtd.setglobalmonitor("use source limits", True)
fdtd.setglobalmonitor("frequency points", 20)

# RCWA region
fdtd.addrcwa()
fdtd.set("name", "RCWA")
fdtd.set("simulation region", "3D")
fdtd.set("x", 0.0); fdtd.set("x span", period)
fdtd.set("y", 0.0); fdtd.set("y span", period)
fdtd.set("z min", -sub_thick - 1e-6); fdtd.set("z max", height + 1e-6)

# KEY: set background material to '<Object defined dielectric>'
print("Setting background material...")
for prop, val in [
    ("background material", "<Object defined dielectric>"),
    ("mesh cells x", 100), ("mesh cells y", 100), ("mesh cells z", 100),
    ("frequency points", 20),
    ("angle theta", 0.0), ("angle phi", 0.0),
    ("incident angle", "single"),
]:
    try:
        fdtd.setnamed("RCWA", prop, val)
        v = fdtd.getnamed("RCWA", prop)
        print(f"  {prop} = {v!r}")
    except Exception as e:
        print(f"  {prop}: FAILED - {str(e)[:60]}")

# Geometry (top level, NOT inside groupscope)
fdtd.addrect()
fdtd.set("name", "substrate")
fdtd.set("x", 0.0); fdtd.set("x span", period)
fdtd.set("y", 0.0); fdtd.set("y span", period)
fdtd.set("z min", -sub_thick); fdtd.set("z max", 0.0)
fdtd.set("material", "SiO2 (Glass) - Palik")

fdtd.addrect()
fdtd.set("name", "pillar")
fdtd.set("x", 0.0); fdtd.set("x span", pillar)
fdtd.set("y", 0.0); fdtd.set("y span", pillar)
fdtd.set("z min", 0.0); fdtd.set("z max", height)
fdtd.set("material", "SiO2 (Glass) - Palik")
print("Geometry added.")

print("\nRunning RCWA...")
t0 = time.perf_counter()
fdtd.run()
dt = time.perf_counter() - t0
print(f"run() in {dt:.2f}s")

z_layers = np.array(fdtd.getresult("RCWA", "z")).flatten() * 1e9
print(f"\nz layers (nm): {z_layers}")
print(f"Grating at {height*1e9:.1f}nm detected: {any(abs(z_layers - height*1e9) < 20)}")

avail = fdtd.getresult("RCWA")
print(f"Available: {avail!r}")

te = fdtd.getresult("RCWA", "total_energy")
lam = np.array(te['lambda']).flatten() * 1e9
Rs  = np.array(te['Rs']).flatten()
Ts  = np.array(te['Ts']).flatten()
sub = fdtd.getresult("RCWA", "substrate")
n_lower = np.array(sub['n_lower']).flatten()[0]
print(f"n_lower (substrate) = {n_lower:.4f}")
print(f"Rs: {Rs[:5]}")
print(f"Rs constant: {np.allclose(Rs, Rs[0])}")

fdtd.close()
print("\nDone.")
