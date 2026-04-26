# Recollida de dades

Esta carpeta contiene la generacion de imagenes, las respuestas de modelos y los resultados derivados del benchmark de colores.

## Estructura

- `scripts/`: funciones compartidas para generar imagenes, consultar modelos, calcular errores y guardar graficos.
- `notebooks/`: notebooks principales de recogida de datos; `notebooks/old/` conserva versiones anteriores.
- `experiments/soroll/`: experimento con imagenes con ruido.
- `experiments/sense-soroll/`: experimento con colores solidos sin ruido.
- `archive/`: temporales antiguos, checkpoints de Jupyter y salidas auxiliares que no son resultados finales.

## Experimentos

Cada experimento mantiene la misma organizacion:

- `images/`: imagenes usadas en las consultas.
- `csv/`: datos base y resultados directos de `gpt-4o` / `gpt-4o-mini`.
- `model-runs/`: salidas crudas de ejecuciones adicionales, como la familia GPT-5 con prompt RGB JSON.
- `results/`: tablas finales combinadas y resumenes.
- `plots/`: graficos finales listos para usar.
- `logs/`: logs de ejecucion.

## CSV principales

En `experiments/soroll/csv/`:

- `input_image_sample.csv`: muestra de 1000 colores usada para las imagenes con ruido.
- `outputmodel_image_sample_4o.csv`: respuestas crudas de `gpt-4o` y `gpt-4o-mini`.
- `sample-colors_4o.csv`: dataset final de los modelos 4o con error cromatico.
- `sample-colors_4o.xlsx`: version Excel del dataset 4o.

En `experiments/sense-soroll/csv/`:

- `input_image_sample.csv`: misma muestra de 1000 colores.
- `outputmodel_image_sample_4o.csv`: respuestas crudas de `gpt-4o` y `gpt-4o-mini` sobre colores solidos.
- `sample-colors_4o.csv`: dataset final de los modelos 4o sobre colores solidos.

## Resultados finales

Con ruido:

- `experiments/soroll/results/sample_colors_tots_models_actual.csv`
- `experiments/soroll/results/resum_error_tots_models.csv`
- `experiments/soroll/results/resum_lazy_response_tots_models.csv`
- `experiments/soroll/results/resum_percent_acierto_tots_models.csv`
- `experiments/soroll/plots/`

Sin ruido:

- `experiments/sense-soroll/results/sample_colors_solids_tots_models_actual.csv`
- `experiments/sense-soroll/results/resum_error_solids_tots_models.csv`
- `experiments/sense-soroll/results/resum_lazy_response_solids_tots_models.csv`
- `experiments/sense-soroll/results/resum_percent_acierto_solids_tots_models.csv`
- `experiments/sense-soroll/plots/`

## Notebooks

- `notebooks/con-ruido.ipynb`: experimento con imagenes con ruido.
- `notebooks/sin-ruido.ipynb`: experimento con colores solidos sin ruido.
- `notebooks/old/`: versiones anteriores y pilotos, incluido el notebook GPT-5 RGB JSON.
