"""
Pipeline: RCWA 2500 -> train ANN -> RCWA +2500 -> retrain ANN on 5000 -> compare
"""
import multiprocessing as mp, pandas as pd, subprocess, time, sys, os

sys.path.insert(0, '.')
from data_generation.generate import _worker, _init_worker, _precompute_epsilon
from tqdm import tqdm

CORES = min(mp.cpu_count()-1, 8)
PARAMS = 'data_generation/output/rcwa_params_train.csv'

def run_rcwa(rows_slice, outfile, label):
    ep = _precompute_epsilon('Si')
    df = pd.read_csv(PARAMS).iloc[rows_slice]
    tasks = [(row.Index, row.w, row.h, row.p) for row in df.itertuples()]
    print(f'\n=== RCWA {label}: {len(tasks)} rows on {CORES} cores ===')
    t0 = time.time()
    results = []
    with mp.Pool(processes=CORES, initializer=_init_worker, initargs=(ep,)) as pool:
        for r in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc=label):
            results.append(r)
    cols = ['id','w','h','p'] + [f'R_wl_{i}' for i in range(100)]
    df_out = pd.DataFrame(results, columns=cols).drop(columns=['id'])
    df_out.to_csv(outfile, index=False)
    print(f'{label} done in {(time.time()-t0)/60:.1f} min -> {outfile} ({len(df_out)} rows)')

def train_ann(data, model, scaler, label):
    print(f'\n=== ANN train {label}: {data} ===')
    r = subprocess.run([
        sys.executable, 'ann/train.py',
        '--data', data, '--model', model, '--scaler', scaler, '--epochs', '300',
    ], capture_output=False)
    if r.returncode != 0:
        print(f'ANN training failed for {label}')

def eval_mae(model_path, scaler_path, label):
    import torch, joblib, numpy as np
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, '.')
    from ann.train import ANN
    from scipy.interpolate import interp1d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load(scaler_path)
    model = ANN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate on test set (500 rows)
    df_test = pd.read_csv('data_generation/output/rcwa_Si_test.csv')
    X = scaler.transform(df_test[['w','h','p']].values)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Y_pred = model(X_t).cpu().numpy()
    Y_true = df_test[[f'R_wl_{i}' for i in range(100)]].values
    mae = float(np.abs(Y_pred - Y_true).mean())
    print(f'\n>>> {label} MAE on test set: {mae*100:.4f}% <<<')
    return mae

if __name__ == '__main__':
    # Batch 1: rows 0-2499
    run_rcwa(slice(0, 2500), 'data_generation/output/rcwa_Si_train_2500.csv', 'batch1')
    train_ann('data_generation/output/rcwa_Si_train_2500.csv',
              'ann/model_2500.pth', 'ann/scaler_2500.pkl', '2500-row')
    mae_2500 = eval_mae('ann/model_2500.pth', 'ann/scaler_2500.pkl', 'ANN@2500')

    # Batch 2: rows 2500-4999
    run_rcwa(slice(2500, 5000), 'data_generation/output/rcwa_Si_train_2500_5000.csv', 'batch2')

    # Combine 5000 rows
    df_all = pd.concat([
        pd.read_csv('data_generation/output/rcwa_Si_train_2500.csv'),
        pd.read_csv('data_generation/output/rcwa_Si_train_2500_5000.csv'),
    ], ignore_index=True)
    df_all.to_csv('data_generation/output/rcwa_Si_train_5000.csv', index=False)
    print(f'\nCombined: {len(df_all)} rows -> rcwa_Si_train_5000.csv')

    train_ann('data_generation/output/rcwa_Si_train_5000.csv',
              'ann/model_5000.pth', 'ann/scaler_5000.pkl', '5000-row')
    mae_5000 = eval_mae('ann/model_5000.pth', 'ann/scaler_5000.pkl', 'ANN@5000')

    print(f'\n========= COMPARISON =========')
    print(f'ANN@2500  MAE = {mae_2500*100:.4f}%')
    print(f'ANN@5000  MAE = {mae_5000*100:.4f}%')
    print(f'Improvement = {(mae_2500-mae_5000)*100:.4f} pp')
