# terminal_A

- 当前保存版本：`3张误杀版`
- 来源提交：`8305cbc`
- 当前参数文件：`recipe.json`

建议使用方式：

```powershell
$env:PYTHONPATH='.vendor;src'
python src\run_batch.py --root . --config recipes\terminal_A\recipe.json --raw datasets\terminal_A\raw --out runs\terminal_A\<run_name>\report.csv --debug runs\terminal_A\<run_name>\debug
```

兼容说明：

- 当前项目仍保留 `data\raw` 作为兼容入口。
- 实际原始图会归档到 `datasets\terminal_A\raw`。
