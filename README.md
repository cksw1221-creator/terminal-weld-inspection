# Terminal Weld Inspection

OpenCV version for terminal weld inspection.

## Directory Layout

```text
recipes/
  terminal_A/
    recipe.json

datasets/
  terminal_A/
    raw/

runs/
  terminal_A/
    2026-04-24_拉长版/
    2026-04-24_3张误杀版/
```

当前 `terminal_A` 参数已保存到：

- `recipes/terminal_A/recipe.json`

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
