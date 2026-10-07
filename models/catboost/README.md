# Modelo CatBoost para DataFest

Este directorio contiene el entrenador de CatBoost y la validación de la
entrega para la competencia DataFest. El modelo estima la probabilidad de que
un cliente convierta (`objetivo = 1`) en el mes de prueba.

## Estructura relevante

```text
datafest/
├── train.csv
├── test.csv
├── sample_submission.csv
├── metaData.csv
├── requirements.txt
├── DATASET_DESCRIPTION.md
├── models/
│   ├── common.py
│   └── catboost/
│       ├── README.md
│       ├── train_model.py
│       ├── validate_submission.py
│       └── submission.csv
```

Los archivos CSV son los archivos oficiales proporcionados para la
competencia y deben permanecer en la raíz del repositorio. La descripción
completa de sus columnas está en
[`DATASET_DESCRIPTION.md`](../../DATASET_DESCRIPTION.md).

## Qué es CatBoost

CatBoost es un algoritmo de *gradient boosting* basado en árboles de decisión.
Está diseñado para trabajar directamente con variables categóricas, sin
necesidad de convertirlas previamente mediante *one-hot encoding*.

Las variables categóricas usadas por este modelo son:

- `ocupacion`
- `region`
- `canal_adquisicion`
- `banda_riesgo`
- `dispositivo_principal`

El modelo utiliza `CatBoostClassifier` y genera probabilidades entre `0` y
`1`, no clases binarias. Por lo tanto, una predicción de `0.80` significa que
el modelo estima una mayor propensión de conversión que una predicción de
`0.20`.

## Requisitos

- Python 3.10 o superior.
- Los archivos de la competencia en la raíz del repositorio.
- Las dependencias definidas en `requirements.txt`.

Para crear un entorno reproducible en Windows PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, se puede ejecutar el
entrenamiento usando directamente
`.venv\Scripts\python.exe`.

## Ejecución

Los comandos deben ejecutarse desde la raíz del repositorio, no desde este
directorio:

```powershell
python -m models.catboost.train_model
```

El comando usa `900` iteraciones como máximo y guarda el resultado en:

```text
models/catboost/submission.csv
```

Para permitir más iteraciones:

```powershell
python -m models.catboost.train_model --iterations 1200
```

Para escribir el resultado en otra ubicación:

```powershell
python -m models.catboost.train_model --output resultados/catboost.csv
```

La carpeta de salida debe existir antes de ejecutar el comando. La ruta
predeterminada sí existe dentro del repositorio.

## Flujo de entrenamiento

El entrenador realiza las siguientes etapas:

1. Carga `train.csv` y `test.csv` desde la raíz del repositorio.
2. Ordena temporalmente las observaciones por `id_cliente` y `mes`.
3. Calcula variables históricas causales en `models/common.py`.
4. Excluye `id_cliente` como predictor directo para evitar que el identificador
   actúe como una variable numérica arbitraria.
5. Reserva el último mes disponible de `train.csv` (`202611`) para validación.
6. Entrena un modelo inicial con `eval_metric="AUC"` y parada temprana.
7. Informa AUC y Gini sobre ese mes de validación.
8. Recupera el número de iteraciones seleccionado y entrena un modelo final con
   todos los registros de entrenamiento.
9. Predice el mes de prueba (`202612`) y guarda la entrega.

La división temporal es preferible a una división aleatoria porque el concurso
evalúa sobre un mes posterior. Así se aproxima mejor al comportamiento real y
se evita evaluar con información de un período futuro.

## Variables históricas y prevención de fuga

Para cada cliente se calculan, antes de la observación actual:

- `hist_observaciones`: cantidad de observaciones anteriores.
- `hist_conversiones`: cantidad acumulada de conversiones anteriores.
- `hist_tasa_conversion`: proporción histórica de conversiones.

Estas variables no incluyen el `objetivo` de la fila actual. Para una fila del
mes de prueba se usa únicamente el historial disponible en `train.csv`. Esto
evita que el modelo use información futura o la etiqueta que intenta predecir.

## Parámetros principales

| Parámetro | Valor | Propósito |
|---|---:|---|
| `iterations` | `900` | Máximo de árboles; se puede cambiar con `--iterations`. |
| `learning_rate` | `0.05` | Tamaño de cada actualización. |
| `depth` | `7` | Profundidad máxima de los árboles. |
| `l2_leaf_reg` | `5` | Regularización L2. |
| `early_stopping_rounds` | `80` | Detiene el entrenamiento si AUC deja de mejorar. |
| `random_seed` | `42` | Hace reproducible el entrenamiento. |
| `thread_count` | `-1` | Usa los hilos disponibles. |
| `allow_writing_files` | `False` | Evita archivos auxiliares de CatBoost. |

## Evaluación

La métrica principal es ROC AUC, porque la competencia evalúa la capacidad de
ordenar correctamente los clientes según su propensión de conversión.

También se informa el coeficiente de Gini:

```text
Gini = 2 * AUC - 1
```

Un valor mayor indica un mejor ordenamiento de las probabilidades. La métrica
de validación que aparece en la consola corresponde únicamente al último mes
de entrenamiento; no es el puntaje oficial del conjunto de prueba.

Una salida esperada tiene esta forma:

```text
Temporal validation month=202611: AUC=0.XXXXXX, Gini=0.XXXXXX
Saved 9,900 predictions to ...
```

## Validar la entrega

Antes de subir el archivo al concurso, ejecutar:

```powershell
python -m models.catboost.validate_submission
```

El validador comprueba que:

- El archivo tenga exactamente las columnas `id_cliente,prediccion`.
- Tenga una fila por cada registro de `test.csv`.
- Los `id_cliente` coincidan con `test.csv` y conserven el orden original.
- No existan predicciones faltantes.
- Todas las predicciones estén entre `0` y `1`.

También se puede validar otro archivo:

```powershell
python -m models.catboost.validate_submission `
  --submission resultados/catboost.csv
```

La entrega final debe conservar `id_cliente` y no debe incluir `mes`. El
archivo de ejemplo `sample_submission.csv` solo muestra la estructura
requerida; sus predicciones no deben reutilizarse.

## Reproducibilidad

Para repetir el experimento:

1. Usar la misma versión de Python y las mismas dependencias.
2. Mantener los CSV originales sin modificar.
3. Ejecutar el comando desde la raíz del repositorio.
4. Mantener `random_seed=42`.
5. Comparar la salida temporal de AUC/Gini y validar el archivo generado.

El entrenamiento no guarda el modelo binario; únicamente genera el CSV de
predicciones. Para una entrega nueva, se debe volver a ejecutar el
entrenamiento y luego validar el archivo.

## Alcance de este directorio

Este README documenta exclusivamente el pipeline de CatBoost. No se asume que
existan implementaciones funcionales de LightGBM, XGBoost o un ensemble en el
estado actual del repositorio. Si se agregan posteriormente, deben tener su
propio entrenador, validación y documentación.
