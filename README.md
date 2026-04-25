# color-llm-benchmark

UPC PE: benchmark para evaluar si distintos LLMs identifican bien el color principal de una imagen.

La idea del proyecto es separar la recogida de datos y el analisis. La recogida se hace en Python, porque genera colores, imagenes y llamadas a API. El analisis se hace en R. Los notebooks explican y ejecutan los pasos principales, pero la logica reutilizable esta en scripts.

## Estructura del proyecto

```text
project-root/
|-- recollida-dades/
|   |-- dades.ipynb
|   |-- scripts/
|   |-- csv/
|   |-- images/
|   `-- logs/
|-- analisis/
|   |-- analisis.ipynb
|   `-- scripts/
|-- docs/
|   |-- inicial/
|   |-- resultados/
|   `-- ppt/
`-- README.md
```

## Instalacion en otro ordenador

Estos pasos estan pensados para Windows y PowerShell. La idea es poder clonar el proyecto en otro PC y reconstruir el entorno desde cero.

### 1. Instalar requisitos del sistema

Instala Git, Python y R si no estan instalados.

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id RProject.R -e
```

Cierra y vuelve a abrir PowerShell despues de instalar. Luego comprueba que todo responde:

```powershell
git --version
python --version
Rscript --version
```

Si `Rscript` no se reconoce, revisa que la carpeta de R este en el `PATH`. En una instalacion normal suele ser:

```text
C:\Program Files\R\R-4.5.3\bin
```

### 2. Clonar el repositorio

```powershell
cd C:\Users\TU_USUARIO\Documents\dev
git clone git@github-adriasebastia:adriasebastia/color-llm-benchmark.git
cd color-llm-benchmark
```

Si no tienes configurado el alias SSH `github-adriasebastia`, puedes usar HTTPS:

```powershell
git clone https://github.com/adriasebastia/color-llm-benchmark.git
```

### 3. Crear el entorno Python para Jupyter

JupyterLab se ejecuta desde un entorno virtual local llamado `.venv`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel numpy pandas pillow scikit-image openai python-dotenv
.\.venv\Scripts\python.exe -m ipykernel install --user --name color-llm-benchmark-py --display-name "Python (color-llm-benchmark)"
```

Esto crea la carpeta `.venv/`, que no se sube a Git porque es solo del ordenador local.

### 3.1. Configurar la API key

Para ejecutar la parte de modelos, crea un fichero `.env` en la raiz del proyecto. No lo subas nunca a Git.

```powershell
Copy-Item .env.example .env
notepad .env
```

Dentro de `.env`, deja la variable asi, pero usando tu clave real:

```text
OPENAI_API_KEY=tu_clave_aqui
```

En el notebook `recollida-dades/dades.ipynb`, las llamadas a modelos estan apagadas por defecto con `RUN_MODEL_QUERIES = False`. Cambialo a `True` solo cuando quieras gastar llamadas de API.

### 4. Instalar paquetes de R

Ejecuta este comando desde la raiz del proyecto:

```powershell
Rscript -e "install.packages(c('tidyverse','magick','jsonlite','httr2','dotenv','IRkernel'), repos='https://cloud.r-project.org')"
```

Si Windows intenta instalar paquetes en `C:\Program Files\R\...` y da error de permisos, abre R una vez y acepta crear una libreria personal de usuario cuando lo pregunte.

### 5. Registrar el kernel de R en Jupyter

Ejecuta:

```powershell
Rscript -e "IRkernel::installspec(name='color-llm-benchmark-r', displayname='R (color-llm-benchmark)', user=TRUE)"
```

Para comprobar que Jupyter ve el kernel:

```powershell
.\.venv\Scripts\jupyter.exe kernelspec list
```

Deberia aparecer algo parecido a:

```text
color-llm-benchmark-r
```

### 6. Abrir JupyterLab

Desde la raiz del proyecto:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

En el navegador, abre los notebooks:

```text
recollida-dades/dades.ipynb
analisis/analisis.ipynb
```

Selecciona el kernel correcto en cada uno:

```text
recollida-dades/dades.ipynb -> Python (color-llm-benchmark)
analisis/analisis.ipynb -> R (color-llm-benchmark)
```

## Ejecucion del proyecto

1. Ejecutar primero:

```text
recollida-dades/dades.ipynb
```

Este notebook prepara carpetas, carga funciones Python desde `recollida-dades/scripts/`, genera el CSV base y escribe logs.
Concretamente genera:

```text
recollida-dades/csv/input_image_sample.csv
recollida-dades/images/*.png
```

El CSV contiene los 1000 colores aleatorios, los canales `r`, `g`, `b`, el hexadecimal y el `chroma`. El `chroma` se calcula en CIELAB como `sqrt(a^2 + b^2)` y se normaliza entre 0 y 1 dividiendo por el maximo posible dentro de sRGB. Las imagenes son PNG de 100x100 pixels con un 60% del color objetivo, 20% de un color cercano y 20% de un color aleatorio.

2. Ejecutar despues:

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
