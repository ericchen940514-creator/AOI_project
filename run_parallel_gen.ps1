# Launch 4 independent single-process generators in parallel, then merge + train
$BASE  = "c:\Users\user\OneDrive\Desktop\AOI_project"
$GEN   = "$BASE\sio2\generate_torcwa_sio2.py"
$OUTDIR = "$BASE\data_generation\output"
$MAIN  = "$OUTDIR\rcwa_sio2_train_ng9.csv"
$TMP   = @("$OUTDIR\tmp_gen_1.csv","$OUTDIR\tmp_gen_2.csv","$OUTDIR\tmp_gen_3.csv","$OUTDIR\tmp_gen_4.csv")
$SEEDS = @(101, 202, 303, 404)
$NROWS = 1300  # each job generates 1300 rows => 4*1300=5200 total new rows

Write-Host "=== Starting 4 parallel generators ===" -ForegroundColor Cyan

$jobs = @()
for ($i = 0; $i -lt 4; $i++) {
    $out  = $TMP[$i]
    $seed = $SEEDS[$i]
    $log  = "$OUTDIR\gen_job_$i.log"
    $proc = Start-Process python -ArgumentList "`"$GEN`"","--nrows","$NROWS","--workers","1","--seed","$seed","--out","`"$out`"" `
        -WorkingDirectory $BASE `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$OUTDIR\gen_job_${i}_err.log" `
        -WindowStyle Normal -PassThru
    $jobs += $proc
    Write-Host "  Job $i started (PID $($proc.Id), seed=$seed, out=$(Split-Path $out -Leaf))"
}

Write-Host "`nWaiting for all 4 jobs to finish..." -ForegroundColor Yellow
$jobs | Wait-Process
Write-Host "All jobs done!" -ForegroundColor Green

# Merge: append data rows (skip header) from each tmp CSV into main CSV
Write-Host "`n=== Merging into main CSV ===" -ForegroundColor Cyan
$appended = 0
foreach ($tmp in $TMP) {
    if (Test-Path $tmp) {
        $lines = Get-Content $tmp | Select-Object -Skip 1  # skip header
        Add-Content -Path $MAIN -Value $lines
        $appended += $lines.Count
        Write-Host "  Merged $(Split-Path $tmp -Leaf): $($lines.Count) rows"
        Remove-Item $tmp -Force
    } else {
        Write-Host "  WARNING: $tmp not found" -ForegroundColor Red
    }
}

$total = (Get-Content $MAIN | Measure-Object -Line).Lines - 1
Write-Host "`nTotal rows in CSV: $total" -ForegroundColor Green

# Start training
Write-Host "`n=== Starting training ===" -ForegroundColor Cyan
python "$BASE\train_ann_sio2.py" --epochs 500 --fresh --smooth 1.0 --smooth-labels

Write-Host "`n=== Evaluating ===" -ForegroundColor Cyan
python "$BASE\eval_ann_sio2.py"
