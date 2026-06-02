"""
Test groupscope: place geometry inside RCWA group so RCWA detects it.
Fresh session (no .fsp load).
"""
import sys, time
sys.path.insert(0, 'C:/Program Files/Lumerical/v241/api/python')
import lumapi, numpy as np

period = 680e-9
pillar = 309.1e-9
height = 209.7e-9
sub_thick = 2e-6

fdtd = lumapi.FDTD(hide=True)
print("Connected (fresh session).")

fdtd.setglobalsource("wavelength start", 400e-9)
fdtd.setglobalsource("wavelength stop",  750e-9)
fdtd.setglobalmonitor("use wavelength spacing", True)
fdtd.setglobalmonitor("use source limits", True)
fdtd.setglobalmonitor("frequency points", 20)

# Try groupscope via eval in fresh session
print("\nTest 1: groupscope via eval (fresh session)")
lsf1 = f"""
addrcwa;
set("name","RCWA");
set("simulation region","3D");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{-sub_thick-1e-6}); set("z max",{height+1e-6});
"""
try:
    fdtd.eval(lsf1)
    print("  addrcwa eval: OK")
except Exception as e:
    print(f"  addrcwa eval: {e}")

# Try groupscope("::model::RCWA")
for gs_path in ["::model::RCWA", "RCWA", "::model:RCWA", "model::RCWA"]:
    lsf_gs = f"""
groupscope("{gs_path}");
addrect;
set("name","substrate");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{-sub_thick}); set("z max",0);
set("material","SiO2 (Glass) - Palik");
groupscope("::model");
"""
    try:
        fdtd.eval(lsf_gs)
        print(f"  groupscope('{gs_path}'): OK!")
        break
    except Exception as e:
        print(f"  groupscope('{gs_path}'): {str(e)[:60]}")

# Also add pillar
lsf_pillar = f"""
groupscope("::model::RCWA");
addrect;
set("name","pillar");
set("x",0); set("x span",{pillar});
set("y",0); set("y span",{pillar});
set("z min",0); set("z max",{height});
set("material","SiO2 (Glass) - Palik");
groupscope("::model");
"""
try:
    fdtd.eval(lsf_pillar)
    print("  pillar groupscope: OK")
except Exception as e:
    print(f"  pillar groupscope: {str(e)[:60]}")

print("\nRunning...")
t0 = time.perf_counter()
fdtd.run()
dt = time.perf_counter() - t0
print(f"run() in {dt:.2f}s")

z_layers = np.array(fdtd.getresult("RCWA", "z")).flatten() * 1e9
print(f"z layers (nm): {z_layers}")
has_grating = any(abs(z_layers - height*1e9) < 20)
print(f"Grating at {height*1e9:.0f}nm detected: {has_grating}")

te = fdtd.getresult("RCWA", "total_energy")
Rs = np.array(te['Rs']).flatten()
lam = np.array(te['lambda']).flatten() * 1e9
print(f"Rs: {Rs[:5]}")
print(f"Rs constant: {np.allclose(Rs, Rs[0])}")

fdtd.close()
print("\nDone.")
