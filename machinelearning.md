Aquí tienes una tabla comparativa de los principales modelos de aprendizaje supervisado:  

| **Modelo**               | **Métricas Importantes**                              | **Explicación y Fórmula**                                             | **Hiperparámetros**                               | **Usos**                                        | **Pasos Previos (EDA y Preprocesamiento)**     |
|--------------------------|------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **Regresión Lineal**     | MSE, RMSE, R²                                       | Modela la relación entre variables dependientes e independientes mediante una ecuación lineal. **Fórmula:**  \( y = \beta_0 + \beta_1x + \epsilon \) | Alpha (regularización)                          | Predicción de valores continuos (precio de casas, ventas) | - Manejo de valores nulos  - Eliminación de outliers  - Normalización/Estandarización  - Análisis de correlación |
| **Regresión Logística**  | Accuracy, Precision, Recall, F1-score, AUC-ROC      | Clasifica datos en dos categorías usando una función sigmoide. **Fórmula:**  \( P(y=1) = \frac{1}{1+e^{-(\beta_0 + \beta_1x)}} \) | C (inversa de la regularización), Solver       | Clasificación binaria (detección de fraudes, enfermedades) | - Balanceo de clases - Conversión de variables categóricas - Normalización de variables numéricas |
| **Árboles de Decisión**  | Accuracy, Gini, Entropía                            | Crea una estructura jerárquica dividiendo los datos en función de características relevantes. **Fórmula:** Gini \( = 1 - \sum p_i^2 \) | Profundidad, Número de nodos, Criterio (Gini/Entropía) | Clasificación y regresión (diagnóstico médico, crédito bancario) | - Manejo de valores nulos - Codificación de variables categóricas |
| **Random Forest**        | Accuracy, RMSE, R², OOB Score                       | Ensamble de múltiples árboles de decisión que mejora la precisión y evita sobreajuste. | Número de árboles, Profundidad máxima, Número de características seleccionadas | Predicción en datos complejos, clasificación y regresión (clima, precios de mercado) | - Conversión de variables categóricas - Manejo de datos faltantes - Normalización |
| **SVM (Máquinas de Soporte Vectorial)** | Accuracy, Precision, Recall, AUC-ROC | Encuentra el hiperplano óptimo para clasificar los datos. **Fórmula:** \( f(x) = w \cdot x + b \) | Kernel (lineal, RBF, polinomial), C, Gamma | Clasificación en datos con fronteras complejas (biometría, detección de spam) | - Estandarización de datos - Conversión de variables categóricas |
| **k-NN (k-Nearest Neighbors)** | Accuracy, Precision, Recall | Asigna una clase a una instancia según sus k vecinos más cercanos. **Distancia Euclidiana:** \( d(p,q) = \sqrt{\sum (q_i - p_i)^2} \) | Número de vecinos (k), Distancia (Euclídea, Manhattan) | Clasificación y regresión (recomendaciones, diagnóstico médico) | - Normalización/Estandarización - Conversión de variables categóricas |
| **Redes Neuronales**     | Accuracy, Loss Function, AUC-ROC                    | Modela relaciones complejas a través de capas de neuronas interconectadas. **Fórmula:** \( y = f(WX + B) \) donde \( f \) es una función de activación | Número de capas, Número de neuronas, Learning Rate, Función de activación | Reconocimiento de imágenes, procesamiento de lenguaje natural | - Normalización - Balanceo de clases - Conversión de variables categóricas - Creación de embeddings |

Aquí tienes las fórmulas de las métricas más importantes utilizadas en aprendizaje supervisado:

---

### **Métricas para Regresión:**
1. **Error Cuadrático Medio (MSE - Mean Squared Error)**
   \[
   MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \]
   - Penaliza los errores grandes de manera cuadrática.

2. **Raíz del Error Cuadrático Medio (RMSE - Root Mean Squared Error)**
   \[
   RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   \]
   - Es la raíz cuadrada del MSE, manteniendo la escala original de los datos.

3. **Coeficiente de Determinación (R² - R-Squared)**
   \[
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   \]
   - Indica qué porcentaje de la varianza de los datos es explicada por el modelo.

---

### **Métricas para Clasificación:**
1. **Precisión (Precision)**
   \[
   Precision = \frac{TP}{TP + FP}
   \]
   - Proporción de verdaderos positivos sobre el total de predicciones positivas.

2. **Sensibilidad o Recall (Tasa de Verdaderos Positivos - TPR)**
   \[
   Recall = \frac{TP}{TP + FN}
   \]
   - Proporción de casos positivos correctamente identificados.

3. **Exactitud (Accuracy)**
   \[
   Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
   \]
   - Proporción de predicciones correctas sobre el total.

4. **Puntuación F1 (F1-score)**
   \[
   F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
   \]
   - Media armónica entre precisión y recall.

5. **Área bajo la curva ROC (AUC-ROC)**
   \[
   AUC = \int_{0}^{1} TPR(FPR) \, d(FPR)
   \]
   - Mide la capacidad del modelo para distinguir entre clases.

---

### **Métricas para Árboles de Decisión y Random Forest:**
1. **Índice de Gini (Gini Index)**
   \[
   Gini = 1 - \sum_{i=1}^{c} p_i^2
   \]
   - Mide la pureza de un nodo en un árbol de decisión.

2. **Entropía (Entropy)**
   \[
   Entropía = -\sum_{i=1}^{c} p_i \log_2 (p_i)
   \]
   - Mide la incertidumbre de un nodo en un árbol de decisión.

---

### **Distancias en k-NN:**
1. **Distancia Euclidiana**
   \[
   d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}
   \]
   - Se usa para encontrar vecinos más cercanos en k-NN.

2. **Distancia de Manhattan**
   \[
   d(p, q) = \sum_{i=1}^{n} |q_i - p_i|
   \]
   - Alternativa a la distancia Euclidiana cuando los datos tienen estructuras en cuadrícula.

---

### **Métricas para Regresión**
1. **Error Cuadrático Medio (MSE - Mean Squared Error)**
   \[
   MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \]
   
2. **Raíz del Error Cuadrático Medio (RMSE - Root Mean Squared Error)**
   \[
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   \]

3. **Coeficiente de Determinación (R² - R-Squared)**
   \[
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   \]

### **Métricas para Clasificación**
1. **Precisión (Precision)**
   \[
   Precision = \frac{TP}{TP + FP}
   \]

2. **Sensibilidad o Recall (Tasa de Verdaderos Positivos - TPR)**
   \[
   Recall = \frac{TP}{TP + FN}
   \]

3. **Exactitud (Accuracy)**
   \[
   Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
   \]

4. **Puntuación F1 (F1-score)**
   \[
   F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
   \]

5. **Área bajo la curva ROC (AUC-ROC)**
   \[
   AUC = \int_{0}^{1} TPR(FPR) \, d(FPR)
   \]

### **Métricas para Árboles de Decisión y Random Forest**
1. **Índice de Gini (Gini Index)**
   \[
   Gini = 1 - \sum_{i=1}^{c} p_i^2
   \]

2. **Entropía (Entropy)**
   \[
   Entropía = -\sum_{i=1}^{c} p_i \log_2 (p_i)
   \]

### **Distancias en k-NN**
1. **Distancia Euclidiana**
   \[
   d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}
   \]

2. **Distancia de Manhattan**
   \[
   d(p, q) = \sum_{i=1}^{n} |q_i - p_i|
   \]






