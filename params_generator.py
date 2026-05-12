import numpy as np
import pandas as pd
from scipy.stats import qmc

def generate_datasets(total_samples, test_ratio=0.025):
    bounds = {'w': (0.05, 0.25), 'h': (0.1, 0.5), 'p': (0.2, 0.6)}
    
    # 為了過濾佔空比，先超額抽樣 3 倍
    initial_samples = total_samples * 3
    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample_points = sampler.random(n=initial_samples)
    
    l_bounds = [bounds['w'][0], bounds['h'][0], bounds['p'][0]]
    u_bounds = [bounds['w'][1], bounds['h'][1], bounds['p'][1]]
    scaled_samples = qmc.scale(sample_points, l_bounds, u_bounds)
    
    df = pd.DataFrame(scaled_samples, columns=['w', 'h', 'p'])
    
    # 物理限制：佔空比 (w/p) 在 0.3 ~ 0.7 之間
    df['fill_factor'] = df['w'] / df['p']
    valid_df = df[(df['fill_factor'] >= 0.3) & (df['fill_factor'] <= 0.7)].copy()
    valid_df = valid_df.drop(columns=['fill_factor']).reset_index(drop=True)
    
    # 確認剩餘數量足夠，並切分資料集
    if len(valid_df) < total_samples:
        raise ValueError("有效樣本數不足，請增加初始抽樣量。")
        
    final_df = valid_df.head(total_samples)
    
    # 切分 RCWA (預訓練) 與 Tidy3D (微調) 資料集
    # 20500 組 -> 500 組給 Tidy3D，20000 組給 RCWA
    num_tidy3d = int(total_samples * test_ratio)
    
    df_tidy3d = final_df.iloc[:num_tidy3d].reset_index(drop=True)
    df_rcwa = final_df.iloc[num_tidy3d:].reset_index(drop=True)
    
    return df_rcwa, df_tidy3d

# 生成 20,500 組參數 (20000 給 RCWA, 500 給 Tidy3D)
df_rcwa, df_tidy3d = generate_datasets(20500, test_ratio=500/20500)

df_rcwa.to_csv("rcwa_params_20k.csv", index=False)
df_tidy3d.to_csv("tidy3d_params_500.csv", index=False)

print(f"已生成 RCWA 參數集: {len(df_rcwa)} 組")
print(f"已生成 Tidy3D 參數集: {len(df_tidy3d)} 組")