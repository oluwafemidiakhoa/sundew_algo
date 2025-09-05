@echo off
set DATA=data\MIT-BIH Arrhythmia Database.csv
set SWEEP=results\sweep_cm.csv
set BESTCSV=results\best_by_counts.csv
set BESTMD=results\best_by_counts.md
set UPDATE=results\updates\2025-09-ecg-mitbih.md

python -m benchmarks.sweep_ecg --csv "%DATA%" --out "%SWEEP%" --preset ecg_v1 --limit 50000
python -m benchmarks.select_best --csv "%SWEEP%" --out-csv "%BESTCSV%" --out-md "%BESTMD%" ^
  --research-md "%UPDATE%" --dataset-name "MIT-BIH Arrhythmia Database" ^
  --dataset-notes "CSV from PhysioNet; ~50k rows; binary abnormal-beat labels; ecg_v1 sweep." ^
  --min-savings 88 --max-fn 9000 --max-fp-rate 0.08 --sort f1,precision --top-n 20 --describe
