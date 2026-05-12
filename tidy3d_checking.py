import tidy3d as td
import tidy3d.web as web
from tidy3d import material_library

# 定義參數 (單位：um)
w = 0.150  # 寬度
h = 0.300  # 高度
p = 0.400  # 週期 (Pitch)

# 1. 材料設定：從資料庫載入結晶矽 (cSi) 
silicon = material_library['cSi']['Green2008']

# 2. 幾何結構定義 
# 矽基底 (Substrate)
substrate = td.Structure(
    geometry=td.Box(center=[0, 0, -5], size=[td.inf, td.inf, 10]),
    medium=silicon
)

# 矽光柵 (Grating)
grating = td.Structure(
    geometry=td.Box(center=[0, 0, h/2], size=[td.inf, w, h]),
    medium=silicon
)

# 3. 光源與監測器設定 
# 平面波光源 (位於上方往下打)
plane_wave = td.PlaneWave(
    source_time=td.GaussianPulse(freq0=5e14, fwidth=2e14), # 涵蓋可見光波段
    size=(td.inf, td.inf, 0),
    center=[0, 0, h + 1.0],
    direction="-",
    pol_angle=0
)

# 反射率監測器 (Flux Monitor)
monitor_r = td.FluxMonitor(
    center=[0, 0, h + 1.5],
    size=[td.inf, td.inf, 0],
    freqs=np.linspace(4e14, 7.5e14, 100),
    name="reflectance"
)

# 4. 封裝 Simulation 物件 
sim = td.Simulation(
    size=[p, p, h + 4.0],  # X, Y 維度與週期 p 對齊
    structures=[substrate, grating],
    sources=[plane_wave],
    monitors=[monitor_r],
    run_time=1e-12,
    # 設定邊界條件：X, Y 為週期性，Z 為 PML 吸收層 
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.periodic(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml()
    )
)