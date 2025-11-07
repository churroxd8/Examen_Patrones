# Diagnóstico y Prevención de Data Leakage

## a) Taxonomía de Data Leakage

El *data leakage* ocurre cuando la información que no estaría disponible en el momento de la predicción (en producción) se utiliza para entrenar o evaluar un modelo. Esto resulta en métricas de rendimiento artificialmente infladas y un fallo catastrófico cuando el modelo se enfrenta a datos del mundo real.

### 1. Leakage por Preprocesamiento (Preprocessing Leakage)

* **Definición Formal:** Ocurre cuando las operaciones de preprocesamiento (ej. escalado, imputación, normalización) se "ajustan" (fit) utilizando datos de todo el conjunto (entrenamiento + validación/test) antes de la división (split). El estimador aprende parámetros globales (como la media o la varianza) del conjunto de validación.
* **Ejemplo Concreto:** Se usa `StandardScaler` y se llama a `fit_transform()` sobre el 100% de los datos `X` antes de ejecutar `train_test_split()`.
* **Problemática:** El modelo en validación se beneficia de conocer la distribución global del conjunto de prueba. Por ejemplo, al normalizar, "sabe" cuál es el valor máximo y mínimo de los datos que se supone que nunca ha visto.
* **Detección:** Revisar el código. El `train_test_split` debe ser una de las primeras líneas. Cualquier operación `fit` o `fit_transform` debe ocurrir *después* del split.
* **Prevención:** Utilizar la clase `Pipeline` de Scikit-learn, que aplica de forma correcta los pasos de preprocesamiento, ajustando los estimadores solo en los datos de entrenamiento (`X_train`) y aplicando la transformación a `X_train` y `X_test`.

### 2. Leakage por Target Encoding (Target Encoding Leakage)

* **Definición Formal:** Ocurre al codificar una variable categórica usando información derivada del *target*. El *leakage* sutil sucede cuando el *encoding* de un registro de entrenamiento se calcula usando el propio *target* de ese registro, o cuando el *encoding* del set de validación se calcula usando el *target* del set de validación.
* **Ejemplo Concreto:** Para predecir `clic_usuario`, se tiene la feature categórica `ciudad`. Se reemplaza `'ciudad'` por la "tasa de clics promedio de esa ciudad" (ej. `target.groupby('ciudad').mean()`). Si esto se hace antes del split, el valor de un registro de validación "conoce" el promedio del target de su propio grupo.
* **Problemática:** La feature de *encoding* está altamente correlacionada con el target por definición, inflando artificialmente la importancia de la variable. El modelo no aprende el patrón, sino que "memoriza" la respuesta promedio.
* **Detección:** Analizar cómo se generan las features categóricas. Verificar si el *encoding* se realiza *dentro* de un *fold* de validación cruzada y si el *encoding* de los datos de validación se basa *exclusivamente* en los datos de entrenamiento.
* **Prevención:** Aplicar el *target encoding* *después* del `train_test_split`. Para una validación cruzada robusta, el *encoding* debe realizarse *dentro* de cada *fold* (es decir, el *encoding* para el *fold* $k$ se calcula usando los datos de los *folds* $k-1$).

### 3. Leakage Temporal (Temporal Leakage)

* **Definición Formal:** Ocurre en series de tiempo o datos dependientes del tiempo cuando los datos de entrenamiento contienen información del "futuro" relativo a los datos que se intentan predecir.
* **Ejemplo Concreto:** Predecir el precio de una acción a las 10:00 AM usando como feature el "precio promedio móvil de las 10:05 AM".
* **Problemática:** El modelo es trivialmente preciso porque se le está dando la respuesta (o un proxy muy fuerte de la misma) en el *input*. En producción, la información futura no existe.
* **Detección:** Auditar todas las *features* temporales. Asegurarse de que cualquier *feature* agregada o de ventana (window) solo use datos *anteriores* al punto de predicción (usar `rolling().shift(1)` en lugar de `rolling()`).
* **Prevención:** Usar una estrategia de validación cruzada temporal (ej. `TimeSeriesSplit` de Scikit-learn) donde los *folds* de entrenamiento siempre son cronológicamente anteriores a los *folds* de validación.

### 4. Leakage por Features Futuras (Future Information Features)

* **Definición Formal:** Una generalización del *leakage* temporal. Se incluyen variables en el modelo que, si bien no son el *target*, son proxies del mismo que no se conocerían en el momento de la predicción.
* **Ejemplo Concreto:** Predecir si un cliente abandonará (`churn`). Se incluye la *feature* `meses_activo_total`. Sin embargo, si un cliente abandona, esta *feature* deja de crecer. El modelo aprende que `meses_activo_total` bajo $\rightarrow$ `churn` alto, pero esta *feature* solo se conoce *después* del evento de *churn*.
* **Problemática:** Se crea una correlación espuria que no tiene poder predictivo real.
* **Detección:** Análisis de *feature importance*. Si una *feature* tiene un poder predictivo desproporcionado (ej. AUC de 0.99 con una sola *feature*), es sospechosa. Preguntar: "¿Esta *feature* estaría disponible, con este valor exacto, en el momento de ejecutar la predicción?".
* **Prevención:** Simulación manual del *pipeline* de predicción. Definir un "punto de corte" temporal (cutoff time) y asegurar que todas las *features* se calculan usando solo datos disponibles *antes* de ese punto.

### 5. Leakage por Duplicación de Registros (Record Duplication)

* **Definición Formal:** Ocurre cuando los mismos registros (o registros casi idénticos) están presentes tanto en el conjunto de entrenamiento como en el de validación/prueba.
* **Ejemplo Concreto:** Un *dataset* de imágenes donde la `imagen_A.jpg` está en `train` y una copia, `imagen_A_copia.jpg`, está en `test`.
* **Problemática:** El modelo es evaluado sobre datos que ya ha "memorizado" durante el entrenamiento, inflando la métrica de generalización.
* **Detección:** Buscar duplicados exactos (IDs) o casi duplicados (hashes, *features* vectorizadas muy similares) entre los conjuntos `train` y `test`.
* **Prevención:** De-duplicar el *dataset* *antes* de realizar cualquier *split*.

### 6. Leakage por Estratificación Incorrecta o Agrupamiento (Incorrect Stratification / Group Leakage)

* **Definición Formal:** Un tipo sutil de duplicación. Ocurre cuando las observaciones no son independientes (IID), sino que pertenecen a "grupos". Si el *split* no respeta estos grupos, el modelo aprende sobre un grupo en *train* y es evaluado sobre otra parte del mismo grupo en *test*.
* **Ejemplo Concreto:** Predecir una enfermedad usando radiografías. El *dataset* contiene 5 imágenes por paciente. Se hace un `train_test_split` aleatorio. El modelo ve 4 imágenes del `Paciente_A` en *train* y 1 imagen del `Paciente_A` en *test*.
* **Problemática:** El modelo aprende a reconocer las características del *paciente* (un *confounder*), en lugar de la enfermedad. La generalización a *pacientes nuevos* (el caso real de producción) será pésima.
* **Detección:** Identificar la "unidad de análisis" (¿es el paciente, el cliente, la sesión?). Verificar si existen IDs de grupo que se repiten entre *train* y *test*.
* **Prevención:** Usar `GroupKFold` o `GroupShuffleSplit`, asegurando que todas las observaciones de un mismo grupo permanezcan en el mismo conjunto (ya sea *train* o *test*).

## b) Metodología para Evitar el Data Leakage (Checklist Avanzado)

Este *checklist* está diseñado para auditar un pipeline de ML y detectar posibles fuentes de *leakage*.

### Fase 1: Análisis del Split y Validación

1.  **[ ] ¿El Split es lo Primero?**
    * Verificar que la separación (`train_test_split` o la creación de *folds* de CV) es la *primera* operación realizada sobre el conjunto de datos completo (X, y).
2.  **[ ] ¿El Esquema de Validación es Correcto?**
    * Si los datos son temporales, ¿se está usando `TimeSeriesSplit` o una división por fecha de corte?
    * Si los datos están agrupados (pacientes, tiendas, usuarios), ¿se está usando `GroupKFold` para asegurar que los grupos no se mezclen entre *train* y *test*?
3.  **[ ] ¿La Estratificación es Válida?**
    * Si se usa estratificación (ej. `stratify=y`), ¿se está aplicando correctamente sobre una variable categórica (como el *target*) y no sobre una variable que *no* estará disponible (ej. un ID de grupo que no se debe mezclar)?

### Fase 2: Auditoría de Preprocesamiento y Features

4.  **[ ] ¿Se usa `Pipeline` de Scikit-learn?**
    * Verificar que todas las operaciones de preprocesamiento (escaladores, imputadores, *encoders*) están encapsuladas dentro de un `sklearn.pipeline.Pipeline`.
5.  **[ ] (Si no se usa Pipeline) ¿Se evita el `fit` en Test?**
    * Auditar manualmente. ¿Se está usando `fit_transform()` o `fit()` en los datos de *entrenamiento*? (Correcto).
    * ¿Se está usando *solo* `transform()` en los datos de *validación/prueba*? (Correcto).
    * ¿Se está usando `fit_transform()` o `fit()` en el conjunto de *prueba* o en el *dataset completo*? (¡INCORRECTO - LEAKAGE!).
6.  **[ ] ¿Cómo se maneja el Target Encoding?**
    * ¿El *encoding* (ej. `TargetEncoder`, `CatBoostEncoder`) se aplica *después* del *split*?
    * Si se usa CV, ¿el *encoding* se realiza *dentro* de cada *fold* para prevenir que la información del *target* del *fold* de validación se filtre al de entrenamiento?
7.  **[ ] ¿Hay Features Temporales con "Futuro"?**
    * Revisar la definición de todas las *features* basadas en tiempo. ¿Alguna ventana móvil (ej. `rolling(window=7).mean()`) olvida aplicar un `shift(1)` para asegurar que solo usa datos pasados?

### Fase 3: Análisis de Correlación y Rendimiento

8.  **[ ] ¿Hay Correlaciones Sospechosamente Altas?**
    * Calcular la correlación (Pearson, Spearman) entre las *features* y el *target*. ¿Alguna es > 0.95?
    * Entrenar un modelo simple (ej. `DecisionTreeClassifier(max_depth=3)`) con *cada feature* de forma individual. ¿Alguna *feature* por sí sola da un AUC > 0.95? Si es así, investigar: es probable que sea un proxy del *target* (Leakage tipo 4).
9.  **[ ] ¿Existen Duplicados entre Conjuntos?**
    * Verificar si hay IDs de registro o hashes de *features* idénticos entre `X_train` y `X_test`.
10. **[ ] ¿El Rendimiento es "Demasiado Bueno para ser Verdad"?**
    * Contexto de negocio: Si se predice *churn* (un problema difícil) con 99% de AUC, es casi seguro que hay *leakage*. El rendimiento debe ser realista comparado con *benchmarks* de la industria.

### Referencias

* Kaggle. (n.d.). Data leakage. Kaggle Learn. Recuperado el 7 de noviembre de 2025, de https://www.kaggle.com/learn/data-leakage
* Scikit-learn. (n.d.). Common pitfalls and recommended practices. scikit-learn documentation. Recuperado el 7 de noviembre de 2025, de https://scikit-learn.org/stable/common_pitfalls.html