$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name color-llm-benchmark-py --display-name "Python (color-llm-benchmark)"

Write-Host "Python environment ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
Write-Host "Start Jupyter with: .\.venv\Scripts\jupyter-lab.exe"
