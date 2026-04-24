# color-llm-benchmark
UPC PE: Benchmark for evaluating LLMs in identifying the main color in an image.

## Notebooks

### Python

Create or refresh the Python environment:

```powershell
.\scripts\setup_python.ps1
```

Start JupyterLab:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Use the `Python (color-llm-benchmark)` kernel with:

- `notebooks/color_llm_benchmark_python.ipynb`

### R

Install R first and make sure `Rscript.exe` is available in `PATH`. Then run:

```powershell
Rscript scripts/setup_r.R
```

Use the `R (color-llm-benchmark)` kernel with:

- `notebooks/color_llm_benchmark_r.ipynb`
