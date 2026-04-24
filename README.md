# color-llm-benchmark

UPC PE: Benchmark para evaluar si distintos LLMs identifican bien el color principal de una imagen.

La idea del proyecto es separar la recogida de datos y el analisis. Los notebooks explican y ejecutan los pasos principales, pero la logica reutilizable esta en scripts de R.

## Estructura

```text
project-root/
├── recollida-dades/
│   ├── dades.ipynb
│   ├── scripts/
│   ├── csv/
│   ├── images/
│   └── logs/
├── analisis/
│   ├── analisis.ipynb
│   └── scripts/
├── docs/
│   ├── inicial/
│   ├── resultados/
│   └── ppt/
└── README.md
```

## Dependencias

Primero instala R. En Windows se puede hacer con:

```powershell
winget install --id RProject.R -e
```

Despues instala los paquetes de R del proyecto:

```r
install.packages(readLines("requirements-r.txt"), repos = "https://cloud.r-project.org")
```

Para usar R desde Jupyter hace falta instalar IRKernel:

```r
install.packages("IRkernel")
IRkernel::installspec()
```

En este repo tambien hay un script que instala las dependencias y registra el kernel con el nombre del proyecto:

```powershell
Rscript scripts/setup_r.R
```

Si `Rscript` no se reconoce, cierra y abre PowerShell o revisa que `C:\Program Files\R\R-4.5.3\bin` este en el `PATH`.

## Ejecucion

1. Abrir JupyterLab desde la raiz del proyecto:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

2. Ejecutar primero:

```text
recollida-dades/dades.ipynb
```

Este notebook prepara carpetas, carga funciones desde `recollida-dades/scripts/`, genera el CSV base y escribe logs.

3. Ejecutar despues:

```text
analisis/analisis.ipynb
```

Este notebook carga los CSVs e imagenes de `recollida-dades/`, usa funciones desde `analisis/scripts/` y deja preparada la evaluacion.

## Logs y reproducibilidad

Los logs de la recogida se guardan en:

```text
recollida-dades/logs/
```

Las rutas se construyen con `file.path()` para evitar depender de una ruta absoluta del ordenador. Si se cambia la estructura de carpetas, seguramente habra que revisar las celdas de configuracion de los notebooks.

La regla general es no poner bloques largos dentro de los notebooks. Si una parte empieza a crecer, se mueve a un script dentro de `recollida-dades/scripts/` o `analisis/scripts/`.
