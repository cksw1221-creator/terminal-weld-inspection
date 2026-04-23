# Terminal Weld Inspection

OpenCV version for terminal weld inspection.

## Run

```powershell
$env:PYTHONPATH='.vendor;src'
python src\run_batch.py --root . --config config\inspection.json --raw data\raw --out results\report.csv --debug results\debug
```

## Outputs

```text
results/report.csv
results/debug/overlay/   original image with ROI box
results/debug/roi/       cropped ROI image
results/debug/review/    predicted NG or zero-area cases
```

## Test

```powershell
$env:PYTHONPATH='.vendor;src'
python -m unittest discover -s tests
```

