import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import grcwa
from scipy.interpolate import interp1d
import os

# ==========================================
# 1. 物理參數與全域設定
# ==========================================
TEST_MODE = False  # 【開關】設定為 True 時只跑前 100 組；設定為 False 時火力全開跑全部！

WL_MIN, WL_MAX = 0.4, 0.7  # 波長範圍 (um)
NUM_WAVELENGTHS = 100      # 光譜切割點數
wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WAVELENGTHS)

EPSILON_AIR = 1.0  # 空氣介電常數

# ==========================================
# 2. 建立矽 (Silicon) 色散模型 (n, k 內插)
# ==========================================
# 數據來源: Aspnes and Studna
si_nk_data = np.array([
    [0.40, 5.587, 0.303],
    [0.45, 4.676, 0.063],
    [0.50, 4.298, 0.045],
    [0.55, 4.085, 0.031],
    [0.60, 3.939, 0.021],
    [0.65, 3.840, 0.015],
    [0.70, 3.774, 0.010]
])

# 建立全域的內插函數
n_interp = interp1d(si_nk_data[:, 0], si_nk_data[:, 1], kind='cubic', fill_value="extrapolate")
k_interp = interp1d(si_nk_data[:, 0], si_nk_data[:, 2], kind='cubic', fill_value="extrapolate")

def get_silicon_epsilon(wavelength_um):
    """根據波長計算並回傳矽的複數介電常數"""
    n_val = n_interp(wavelength_um)
    k_val = k_interp(wavelength_um)
    complex_n = n_val + 1j * k_val
    return complex_n ** 2

# ==========================================
# 3. 核心運算函數 (Worker)
# ==========================================
def run_rcwa_simulation(params):
    """
    執行單一組 (w, h, p) 的寬頻 RCWA 模擬
    """
    idx, w, h, p = params
    reflectances = []
    
    # grcwa 的運算引擎設定
    grcwa.set_backend('numpy') 
    
    # 【關鍵修正 1】設定光柵週期性的晶格向量 (Lattice Vectors)
    L1 = [p, 0]
    L2 = [0, p]
    
    for wl in wavelengths:
        freq = 1.0 / wl
        ep_si_dynamic = get_silicon_epsilon(wl)
        
        # 【關鍵修正 2】正確的 grcwa.obj 初始化語法 (nG=11)
        obj = grcwa.obj(11, L1, L2, freq, theta=0, phi=0, verbose=0) 
        
        # --- 建立結構 ---
        # 1. 頂部空氣層 (厚度為 0 代表無限厚)
        obj.Add_LayerUniform(0, EPSILON_AIR)
        
        # 2. 中間光柵層 (厚度 h, 將平面網格切分為 50x50)
        Nx, Ny = 50, 50
        obj.Add_LayerGrid(h, Nx, Ny)
        
        # 3. 底部基板 (無限厚矽基板)
        obj.Add_LayerUniform(0, ep_si_dynamic)
        
        # 【關鍵修正 3】必須先呼叫 Init_Setup() 才能餵入圖案
        obj.Init_Setup()
        
        # --- 設定光柵圖案的介電常數矩陣 ---
        ep_grid = np.ones((Nx, Ny), dtype=complex) * EPSILON_AIR
        fill_factor = w / p
        si_pixels = int(Nx * fill_factor)
        ep_grid[:, :si_pixels] = ep_si_dynamic 
        
        # 【關鍵修正 4】將 2D 矩陣攤平為 1D 陣列後餵入
        obj.GridLayer_geteps(ep_grid.flatten())
        
        # 【關鍵修正 5】設定入射光源為 TE波 (s-polarized)
        obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
        
        # --- 求解零階反射率 ---
        R, T = obj.RT_Solve(normalize=1) 
        reflectances.append(R)
        
    return [idx, w, h, p] + reflectances

# ==========================================
# 4. 主程式 (資料讀取與多核心派發)
# ==========================================
if __name__ == '__main__':
    input_csv = "rcwa_params_20k.csv"
    
    # 檢查檔案是否存在
    if not os.path.exists(input_csv):
        print(f"錯誤：找不到檔案 '{input_csv}'。請確定先執行了 LHS 抽樣腳本產生此檔案！")
        exit()
        
    print(f"讀取參數檔案: {input_csv} ...")
    df_params = pd.read_csv(input_csv)
    
    # 測試模式控制
    if TEST_MODE:
        print("\n【注意】目前為測試模式 (TEST_MODE = True)，僅抽取前 100 組進行運算。")
        df_params = df_params.head(100)
    else:
        print("\n🔥 火力全開模式！準備執行全部資料運算。")
    
    # 準備任務清單
    tasks = [(row.Index, row.w, row.h, row.p) for row in df_params.itertuples()]
    
    cores_to_use = max(1, mp.cpu_count() - 1)
    print(f"準備執行 {len(tasks)} 組模擬...")
    print(f"啟動多核心平行運算 (使用核心數: {cores_to_use}) ...\n")
    
    results = []
    
    # 啟動進程池並顯示進度條
    with mp.Pool(processes=cores_to_use) as pool:
        for res in tqdm(pool.imap(run_rcwa_simulation, tasks), total=len(tasks), desc="RCWA 模擬進度"):
            results.append(res)
            
    # ==========================================
    # 5. 整理與匯出數據
    # ==========================================
    # 建立欄位名稱
    columns = ['id', 'w', 'h', 'p'] + [f'R_wl_{i}' for i in range(NUM_WAVELENGTHS)]
    
    df_results = pd.DataFrame(results, columns=columns)
    df_results = df_results.drop(columns=['id']) # 移除輔助用的 id
    
    # 依據模式決定輸出的檔名
    output_csv = "rcwa_spectra_TEST.csv" if TEST_MODE else "rcwa_spectra_pretrain.csv"
    
    df_results.to_csv(output_csv, index=False)
    print(f"\n✅ 運算大功告成！光譜資料已儲存至 {output_csv}")