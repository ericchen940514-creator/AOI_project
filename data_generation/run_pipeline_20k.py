"""
Pipeline: RCWA rows 5000-19999 (15k) -> combine with existing 5k -> train ANN@20k
          -> compare ANN@2500 / ANN@5000 / ANN@20000 vs tidy3d
"""
import multiprocessing as mp, pandas as pd, subprocess, time, sys, os
import numpy as np

sys.path.insert(0, '.')
from data_generation.generate import _worker, _init_worker, _precompute_epsilon
from tqdm import tqdm

CORES  = min(mp.cpu_count()-1, 8)
PARAMS = 'data_generation/output/rcwa_params_train.csv'

def run_rcwa(rows_slice, outfile, label):
    ep = _precompute_epsilon('Si')
    df = pd.read_csv(PARAMS).iloc[rows_slice]
    tasks = [(row.Index, row.w, row.h, row.p) for row in df.itertuples()]
    print(f'\n=== RCWA {label}: {len(tasks)} rows on {CORES} cores ===', flush=True)
    t0 = time.time()
    results = []
    with mp.Pool(processes=CORES, initializer=_init_worker, initargs=(ep,)) as pool:
        for r in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc=label):
            results.append(r)
    cols = ['id','w','h','p'] + [f'R_wl_{i}' for i in range(100)]
    df_out = pd.DataFrame(results, columns=cols).drop(columns=['id'])
    df_out.to_csv(outfile, index=False)
    print(f'{label} done in {(time.time()-t0)/60:.1f} min -> {outfile} ({len(df_out)} rows)', flush=True)

def train_ann(data, model, scaler, label, epochs=300):
    print(f'\n=== ANN train {label}: {data} ===', flush=True)
    r = subprocess.run([
        sys.executable, 'ann/train.py',
        '--data', data, '--model', model, '--scaler', scaler, '--epochs', str(epochs),
    ], capture_output=False)
    if r.returncode != 0:
        print(f'ANN training failed for {label}', flush=True)

def eval_mae(model_path, scaler_path, label):
    import torch, joblib
    from ann.train import ANN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load(scaler_path)
    model = ANN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    df_test = pd.read_csv('data_generation/output/rcwa_Si_test.csv')
    X = scaler.transform(df_test[['w','h','p']].values)
    with torch.no_grad():
        Y_pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    Y_true = df_test[[f'R_wl_{i}' for i in range(100)]].values
    mae = float(np.abs(Y_pred - Y_true).mean())
    print(f'\n>>> {label} MAE on test set: {mae*100:.4f}% <<<', flush=True)
    return mae

def make_comparison_plot(mae_20k):
    import torch, joblib, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
    from scipy.interpolate import interp1d
    from ann.train import ANN

    wavelengths = np.linspace(0.4, 0.7, 100) * 1000  # nm

    def load_and_predict(pth, pkl, w, h, p):
        device = torch.device('cpu')
        m = ANN(); m.load_state_dict(torch.load(pth, map_location=device, weights_only=False)); m.eval()
        s = joblib.load(pkl)
        x = s.transform([[w, h, p]])
        with torch.no_grad():
            return m(torch.tensor(x, dtype=torch.float32)).numpy().flatten()

    # tidy3d reference
    td = pd.read_csv('tidy3d/output/tidy3d_w150_h300_p400.csv')
    td_fn = interp1d(td['wavelength_nm'].values, td['reflectance_fdtd'].values,
                     kind='linear', fill_value='extrapolate')
    td_on_grid = td_fn(wavelengths)

    # RCWA closest
    df_test = pd.read_csv('data_generation/output/rcwa_Si_test.csv')
    df_test['dist'] = ((df_test['w']-0.150)**2 + (df_test['h']-0.300)**2 + (df_test['p']-0.400)**2)**0.5
    best = df_test.nsmallest(1, 'dist').iloc[0]
    rcwa_R = best[[f'R_wl_{i}' for i in range(100)]].values.astype(float)

    # ANN predictions at exact tidy3d geometry
    r2500  = load_and_predict('ann/model_2500.pth',  'ann/scaler_2500.pkl',  0.150, 0.300, 0.400)
    r5000  = load_and_predict('ann/model_5000.pth',  'ann/scaler_5000.pkl',  0.150, 0.300, 0.400)
    r20000 = load_and_predict('ann/model_20000.pth', 'ann/scaler_20000.pkl', 0.150, 0.300, 0.400)

    mae_rcwa  = np.mean(np.abs(rcwa_R  - td_on_grid)) * 100
    mae_2500  = np.mean(np.abs(r2500   - td_on_grid)) * 100
    mae_5000  = np.mean(np.abs(r5000   - td_on_grid)) * 100
    mae_20000 = np.mean(np.abs(r20000  - td_on_grid)) * 100

    print(f'\nMAE vs tidy3d:  RCWA={mae_rcwa:.2f}%  ANN@2500={mae_2500:.2f}%  ANN@5000={mae_5000:.2f}%  ANN@20000={mae_20000:.2f}%', flush=True)

    colors = {'tidy3d':'#222222','rcwa':'#2196F3','ann2500':'#FF9800','ann5000':'#4CAF50','ann20000':'#E91E63'}

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # Panel A: spectra
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(td['wavelength_nm'], td['reflectance_fdtd'], color=colors['tidy3d'], lw=2.5, label='tidy3d FDTD (ref)', zorder=5)
    ax0.plot(wavelengths, rcwa_R,  color=colors['rcwa'],    lw=1.8, ls='--',  label=f'RCWA closest (MAE={mae_rcwa:.2f}%)')
    ax0.plot(wavelengths, r2500,   color=colors['ann2500'], lw=1.5, ls='-.',  label=f'ANN@2500  (MAE={mae_2500:.2f}%)')
    ax0.plot(wavelengths, r5000,   color=colors['ann5000'], lw=1.5, ls=':',   label=f'ANN@5000  (MAE={mae_5000:.2f}%)')
    ax0.plot(wavelengths, r20000,  color=colors['ann20000'],lw=2.0, ls='-',   label=f'ANN@20000 (MAE={mae_20000:.2f}%)')
    ax0.set_xlabel('Wavelength (nm)', fontsize=11)
    ax0.set_ylabel('Reflectance', fontsize=11)
    ax0.set_title('Si grating  W=150 nm  H=300 nm  P=400 nm  —  all methods vs tidy3d', fontsize=12)
    ax0.legend(fontsize=10); ax0.set_xlim(400,700); ax0.set_ylim(0,1); ax0.grid(alpha=0.3)

    # Panel B: residuals
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axhline(0, color='k', lw=0.8)
    ax1.plot(wavelengths, (rcwa_R  - td_on_grid)*100, color=colors['rcwa'],    lw=1.5, label=f'RCWA  {mae_rcwa:.2f}%')
    ax1.plot(wavelengths, (r2500   - td_on_grid)*100, color=colors['ann2500'], lw=1.5, label=f'ANN@2500  {mae_2500:.2f}%')
    ax1.plot(wavelengths, (r5000   - td_on_grid)*100, color=colors['ann5000'], lw=1.5, label=f'ANN@5000  {mae_5000:.2f}%')
    ax1.plot(wavelengths, (r20000  - td_on_grid)*100, color=colors['ann20000'],lw=1.5, label=f'ANN@20000 {mae_20000:.2f}%')
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Residual (R − tidy3d) [%]', fontsize=11)
    ax1.set_title('Residuals vs tidy3d', fontsize=11)
    ax1.legend(fontsize=9); ax1.set_xlim(400,700); ax1.grid(alpha=0.3)

    # Panel C: MAE bar chart (both test-set MAE and vs-tidy3d MAE)
    ax2 = fig.add_subplot(gs[1, 1])
    labels_bar = ['RCWA\n(closest)', 'ANN\n@2500', 'ANN\n@5000', 'ANN\n@20000']
    maes_td    = [mae_rcwa, mae_2500, mae_5000, mae_20000]
    bar_colors = [colors['rcwa'], colors['ann2500'], colors['ann5000'], colors['ann20000']]
    bars = ax2.bar(labels_bar, maes_td, color=bar_colors, width=0.5, edgecolor='k', linewidth=0.7)
    for bar, v in zip(bars, maes_td):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('MAE vs tidy3d (%)', fontsize=11)
    ax2.set_title('MAE vs tidy3d FDTD', fontsize=11)
    ax2.set_ylim(0, max(maes_td)*1.3); ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('ANN surrogate accuracy scaling  —  Si grating vs tidy3d FDTD', fontsize=13, fontweight='bold', y=1.01)
    outpath = 'ann/comparison_vs_tidy3d_20k.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {outpath}', flush=True)

if __name__ == '__main__':
    # Step 1: RCWA rows 5000-19999 (15,000 rows)
    run_rcwa(slice(5000, 20000), 'data_generation/output/rcwa_Si_train_5000_20000.csv', 'batch3_15k')

    # Step 2: Combine all 20,000 rows
    df_all = pd.concat([
        pd.read_csv('data_generation/output/rcwa_Si_train_5000.csv'),
        pd.read_csv('data_generation/output/rcwa_Si_train_5000_20000.csv'),
    ], ignore_index=True)
    df_all.to_csv('data_generation/output/rcwa_Si_train_20000.csv', index=False)
    print(f'\nCombined: {len(df_all)} rows -> rcwa_Si_train_20000.csv', flush=True)

    # Step 3: Train ANN@20000
    train_ann('data_generation/output/rcwa_Si_train_20000.csv',
              'ann/model_20000.pth', 'ann/scaler_20000.pkl', '20000-row', epochs=300)
    mae_20k = eval_mae('ann/model_20000.pth', 'ann/scaler_20000.pkl', 'ANN@20000')

    # Step 4: Comparison plot
    make_comparison_plot(mae_20k)

    print(f'\n========= FINAL COMPARISON =========')
    print(f'ANN@2500   MAE (test) = 1.9897%')
    print(f'ANN@5000   MAE (test) = 1.5552%')
    print(f'ANN@20000  MAE (test) = {mae_20k*100:.4f}%')
    print(f'Plot saved: ann/comparison_vs_tidy3d_20k.png', flush=True)
