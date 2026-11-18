# Guía Definitiva de Machine Learning: Modelos Supervisados, No Supervisados y Control de Ajuste

Esta guía presenta una recopilación detallada de los principales algoritmos de **Machine Learning Supervisado** y **No Supervisado**. Para cada modelo se incluye una explicación conceptual, un ejemplo práctico con código en Python (`scikit-learn`) y la interpretación de sus resultados. Asimismo, se abordan las técnicas fundamentales para diagnosticar y controlar el sobreajuste (*overfitting*) y el subreajuste (*underfitting*).

---

## PARTE 1: Aprendizaje Supervisado
El aprendizaje supervisado utiliza datos etiquetados (pares de entrada $X$ y salida o etiqueta $y$) para entrenar al modelo a predecir la salida correcta para nuevas entradas. Se divide principalmente en dos tareas: **Regresión** (cuando la salida es continua) y **Clasificación** (cuando la salida es categórica o discreta).

---

### 1. Regresión Lineal (Supervisado - Regresión)

#### Explicación Conceptual
Es el modelo más básico y fundamental para predecir un valor continuo. Asume una relación lineal entre las variables independientes (características, $X$) y la variable dependiente ($y$). La ecuación matemática general se expresa como:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$
Donde $w_0$ es el sesgo (*intercept*) y $w_i$ son los pesos o coeficientes que el algoritmo optimiza mediante el método de **Mínimos Cuadrados Ordinarios (OLS)** para minimizar la suma de los errores cuadráticos residuales.

#### Ejemplo Práctico en Python
Supongamos que queremos predecir el precio de una vivienda en función de su superficie en metros cuadrados.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Generar datos sintéticos
np.random.seed(42)
X = 2.5 * np.random.randn(100, 1) + 10  # Superficie en m² (escalada)
y = 3.5 * X.squeeze() + 10 + np.random.randn(100) * 3  # Precio en miles $

# 2. División de datos (Entrenamiento y Prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenamiento del modelo
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# 4. Predicción y Evaluación
y_pred = modelo_lr.predict(X_test)
print(f"Coeficiente (Pendiente): {modelo_lr.coef_[0]:.2f}")
print(f"Intercepto: {modelo_lr.intercept_:.2f}")
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coeficiente de Determinación (R²): {r2_score(y_test, y_pred):.2f}")
```

---

### 2. Regresión Logística (Supervisado - Clasificación Binaria)

#### Explicación Conceptual
A pesar de llevar la palabra "Regresión" en su nombre, es un algoritmo de **clasificación**. Modela la probabilidad de que una instancia pertenezca a una clase particular (por ejemplo, clase 1 o clase 0). Utiliza la función logística o **Sigmoide** para transformar cualquier valor real en un rango entre 0 y 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Donde $z = w^T X + b$. Si el valor resultante es $\geq 0.5$, se clasifica como 1; de lo contrario, como 0.

#### Ejemplo Práctico en Python
Clasificar si un cliente comprará un producto basándose en su edad y tiempo en la web.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generar datos de clasificación binaria
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entrenamiento
modelo_log = LogisticRegression()
modelo_log.fit(X_train, y_train)

# 3. Evaluación
y_pred = modelo_log.predict(X_test)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
```

---

### 3. Árboles de Decisión (Supervisado - Clasificación y Regresión)

#### Explicación Conceptual
Los árboles de decisión toman decisiones dividiendo recursivamente el espacio de características en regiones rectangulares basándose en reglas de decisión simples (preguntas de tipo "si/no"). La selección de las variables de división se realiza maximizando métricas como la **Ganancia de Información** basada en la **Entropía** o la **Impureza de Gini**.

#### Ejemplo Práctico en Python

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Cargar datos de flores Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 2. Entrenamiento del Árbol
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_train, y_train)

# 3. Evaluación
y_pred = arbol.predict(X_test)
print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
```

---

### 4. Support Vector Machines - SVM (Supervisado - Clasificación)

#### Explicación Conceptual
Las máquinas de vectores de soporte buscan encontrar un hiperplano óptimo en un espacio n-dimensional que separe las clases maximizando el **margen** (la distancia entre el hiperplano y los puntos más cercanos de cada clase, llamados **vectores de soporte**). Mediante el uso de **Kernels** (como el Kernel RBF o polinómico), SVM puede transformar datos linealmente inseparables a espacios de mayor dimensión.

#### Ejemplo Práctico en Python

```python
from sklearn.svm import SVC

# Usamos los mismos datos de clasificación binaria anteriores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento con Kernel RBF
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
modelo_svm.fit(X_train, y_train)

y_pred = modelo_svm.predict(X_test)
print(f"Exactitud SVM: {accuracy_score(y_test, y_pred):.2f}")
```

---

### 5. Random Forest (Supervisado - Ensemble / Ensamblaje)

#### Explicación Conceptual
Es una técnica de **Bagging** (Bootstrap Aggregating) que construye múltiples árboles de decisión independientes durante el entrenamiento (entrenados con subconjuntos aleatorios de datos y características) y promedia sus resultados (en regresión) o realiza una votación mayoritaria (en clasificación). Esto reduce drásticamente la varianza y previene el sobreajuste típico de un solo árbol.

#### Ejemplo Práctico en Python

```python
from sklearn.ensemble import RandomForestClassifier

# 1. Entrenamiento
rf_modelo = RandomForestClassifier(n_estimators=100, random_state=42)
rf_modelo.fit(X_train, y_train)

# 2. Evaluación
y_pred = rf_modelo.predict(X_test)
print(f"Exactitud Random Forest: {accuracy_score(y_test, y_pred):.2f}")
```

---

## PARTE 2: Aprendizaje No Supervisado
El aprendizaje no supervisado analiza datos que **no tienen etiquetas**. Su objetivo es descubrir estructuras ocultas, patrones intrínsecos, agrupamientos naturales o reducir la dimensionalidad de los datos.

---

### 1. K-Means (No Supervisado - Agrupamiento / Clustering)

#### Explicación Conceptual
Agrupa los datos en $K$ clústeres distintos basándose en la similitud de sus características. El algoritmo funciona iterativamente:
1. Inicializa aleatoriamente $K$ centroides.
2. Asigna cada punto de datos al centroide más cercano (usando la distancia euclidiana).
3. Recalcula los centroides como la media de todos los puntos asignados a cada clúster.
4. Repite los pasos 2 y 3 hasta que los centroides dejen de moverse.

#### Ejemplo Práctico en Python

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generar datos sin etiquetas
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# 2. Aplicar K-Means con K=4
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print("Coordenadas de los Centroides:\n", kmeans.cluster_centers_[:2])
```

---

### 2. Análisis de Componentes Principales - PCA (No Supervisado - Reducción de Dimensionalidad)

#### Explicación Conceptual
PCA transforma un conjunto de datos con muchas variables correlacionadas en un conjunto más pequeño de variables incorrelacionadas llamadas **componentes principales**. Esto se logra identificando las direcciones de máxima varianza (autovectores de la matriz de covarianza) y proyectando los datos sobre ellas, minimizando la pérdida de información.

#### Ejemplo Práctico en Python

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# 1. Cargar datos de alta dimensión (30 características)
cancer = load_breast_cancer()
X = cancer.data

# 2. Reducir de 30 dimensiones a 2 componentes principales
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Forma original: {X.shape}")
print(f"Forma reducida: {X_reduced.shape}")
print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
```

---

### 3. DBSCAN (No Supervisado - Clustering Basado en Densidad)

#### Explicación Conceptual
A diferencia de K-Means, DBSCAN no requiere especificar el número de clústeres de antemano. Agrupa puntos que están densamente empaquetados, identificando como **ruido** (*outliers*) a aquellos puntos que se encuentran en regiones de baja densidad. Utiliza dos parámetros clave: `eps` (radio de vecindad) y `min_samples` (número mínimo de puntos para formar una región densa).

#### Ejemplo Práctico en Python

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 1. Generar datos con forma de lunas (no lineales)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# 2. Aplicar DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

print(f"Etiquetas únicas asignadas (incluyendo -1 para ruido): {set(y_dbscan)}")
```

---

## PARTE 3: Control del Sobreajuste y Subreajuste

En Machine Learning, el comportamiento del modelo frente a los datos se divide en tres escenarios principales:

1. **Subreajuste (*Underfitting*):** El modelo es demasiado simple (alta tendencia o *bias*), por lo que no logra capturar los patrones subyacentes ni en el conjunto de entrenamiento ni en el de prueba.
2. **Sobreajuste (*Overfitting*):** El modelo es demasiado complejo (alta varianza), memoriza el ruido y los detalles específicos del conjunto de entrenamiento en lugar de generalizar, fallando estrepitosamente en datos nuevos.
3. **Buen Ajuste (*Good Fit*):** El modelo generaliza de manera óptima los patrones subyacentes.

---

### Métodos de Control y Prevención

#### A. Regularización $L_1$ (Lasso) y $L_2$ (Ridge)
Añaden un término de penalización a la función de pérdida para evitar que los pesos crezcan demasiado o volverse innecesarios:
* **Ridge ($L_2$):** Penaliza el cuadrado de los coeficientes ($\lambda \sum w_i^2$), encogiéndolos hacia cero sin eliminarlos por completo.
* **Lasso ($L_1$):** Penaliza el valor absoluto de los coeficientes ($\lambda \sum |w_i|$), lo que puede llevar algunos pesos exactamente a cero, realizando **selección automática de características**.

```python
from sklearn.linear_model import Ridge

# Modelo de Regresión con Regularización L2
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
```

#### B. Validación Cruzada (*Cross-Validation*)
Divide el conjunto de entrenamiento en $K$ partes (pliegues o *folds*). El modelo se entrena en $K-1$ partes y se evalúa en la parte restante, repitiendo este proceso $K$ veces. Esto asegura que la evaluación del modelo no dependa de una partición afortunada o desafortunada de los datos.

```python
from sklearn.model_selection import cross_val_score

# Validación cruzada de 5 pliegues para un modelo de regresión lineal
scores = cross_val_score(modelo_lr, X, y, cv=5, scoring='r2')
print(f"Puntuaciones R² en validación cruzada: {scores}")
print(f"Promedio R²: {scores.mean():.2f}")
```

#### C. Control de Complejidad del Modelo (Hiperparámetros)
* Limitar la profundidad máxima (`max_depth`) en los árboles de decisión.
* Aumentar el número de muestras mínimas por hoja (`min_samples_leaf`).
* Utilizar estrategias de parada temprana (*Early Stopping*) en modelos iterativos como Redes Neuronales o Gradient Boosting cuando la pérdida en validación deja de disminuir.

---

### Métricas de Control y Diagnóstico

Para detectar si un modelo sufre de sobreajuste o subreajuste, se deben comparar siempre las métricas obtenidas en el **conjunto de entrenamiento** frente al **conjunto de prueba (o validación)**:

1. **Curvas de Aprendizaje (*Learning Curves*):** Gráficas que muestran la métrica de error (o rendimiento) en entrenamiento y validación en función del tamaño del conjunto de entrenamiento o de la complejidad del modelo.
   * *Diagnóstico de Underfitting:* El error en entrenamiento y en validación son ambos elevados.
   * *Diagnóstico de Overfitting:* El error en entrenamiento es muy bajo (cercano a cero), pero el error en validación es alto y diverge del de entrenamiento.

2. **Métricas de Evaluación Estándar:**
   * **Para Regresión:**
     * **MSE (Mean Squared Error):** Penaliza errores grandes al elevarlos al cuadrado.
     * **RMSE (Root Mean Squared Error):** En las mismas unidades de la variable objetivo.
     * **$R^2$ Score:** Porcentaje de variabilidad de la variable objetivo explicada por el modelo (idealmente cercano a 1.0).
   * **Para Clasificación:**
     * **Accuracy (Exactitud):** Proporción de predicciones correctas totales.
     * **Precision, Recall y F1-Score:** Fundamentales cuando existen desequilibrios de clases.
     * **Curva ROC y AUC (Area Under Curve):** Evalúa la capacidad discriminativa del modelo en diferentes umbrales de probabilidad.