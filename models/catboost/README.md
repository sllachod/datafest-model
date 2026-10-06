# Modelo CatBoost para predicción de conversiones

## Qué es CatBoost

CatBoost es un algoritmo de *gradient boosting* basado en árboles de decisión.
Está diseñado para trabajar especialmente bien con variables categóricas, que
pueden pasarse al modelo sin convertirlas previamente a one-hot encoding. En
este proyecto las variables categóricas son `ocupacion`, `region`,
`canal_adquisicion`, `banda_riesgo` y `dispositivo_principal`.

El entrenamiento utiliza `CatBoostClassifier` para estimar la probabilidad de
que cada cliente alcance el objetivo (`objetivo`). La salida es una probabilidad
entre `0` y `1`, no una clase fija.

## Estrategia de entrenamiento

1. Se cargan `train.csv` y `test.csv` desde la raíz del proyecto.
2. Se crean variables históricas por `id_cliente`, usando únicamente
   observaciones de meses anteriores para evitar fuga de información.
3. Se reserva el último mes de entrenamiento como validación temporal.
4. Se entrena un primer modelo con `eval_metric="AUC"` y parada temprana.
5. Se recupera el número óptimo de iteraciones y se entrena un modelo final con
   todos los datos de entrenamiento.
6. Se guardan las predicciones en
   `models/catboost/submission.csv`, salvo que se indique otra ruta con
   `--output`.

La métrica AUC mide la capacidad de ordenar clientes positivos por encima de
los negativos. También se informa Gini, calculado como `2 * AUC - 1`.

## Ejecución

Desde la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
python -m models.catboost.train_model
```

Para aumentar el máximo de iteraciones:

```powershell
python -m models.catboost.train_model --iterations 1200
```

El resultado queda en `models/catboost/submission.csv`. El archivo contiene
exactamente las columnas `id_cliente` y `prediccion`, donde `prediccion` es una
probabilidad entre `0` y `1`. Se puede indicar otra ubicación con `--output`:

```powershell
python -m models.catboost.train_model --output resultados/catboost.csv
```

## Parámetros principales

- `learning_rate=0.05`: tamaño de cada actualización.
- `depth=7`: profundidad máxima de los árboles.
- `l2_leaf_reg=5`: regularización L2.
- `early_stopping_rounds=80`: detiene el ajuste si AUC no mejora.
- `random_seed=42`: hace reproducible el entrenamiento.
- `allow_writing_files=False`: evita que CatBoost cree archivos auxiliares.
