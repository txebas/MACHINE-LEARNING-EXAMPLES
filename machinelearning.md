Aquí tienes la tabla con las fórmulas en un formato que puedes copiar y pegar directamente en un documento sin problemas.  

---

### **Comparación de Modelos de Aprendizaje Supervisado**

| **Modelo**               | **Métricas Importantes**                              | **Explicación y Fórmula**                                             | **Hiperparámetros**                               | **Usos**                                        | **Pasos Previos (EDA y Preprocesamiento)**     |
|--------------------------|------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **Regresión Lineal**     | MSE, RMSE, R²                                       | Modela la relación entre variables dependientes e independientes mediante una ecuación lineal. **Fórmula:**  y = β0 + β1x + ε | Alpha (regularización)                          | Predicción de valores continuos (precio de casas, ventas) | - Manejo de valores nulos  - Eliminación de outliers  - Normalización/Estandarización  - Análisis de correlación |
| **Regresión Logística**  | Accuracy, Precision, Recall, F1-score, AUC-ROC      | Clasifica datos en dos categorías usando una función sigmoide. **Fórmula:**  P(y=1) = 1 / (1 + e^-(β0 + β1x)) | C (inversa de la regularización), Solver       | Clasificación binaria (detección de fraudes, enfermedades) | - Balanceo de clases - Conversión de variables categóricas - Normalización de variables numéricas |
| **Árboles de Decisión**  | Accuracy, Gini, Entropía                            | Crea una estructura jerárquica dividiendo los datos en función de características relevantes. **Fórmula de Gini:** Gini = 1 - Σ(pi²) | Profundidad, Número de nodos, Criterio (Gini/Entropía) | Clasificación y regresión (diagnóstico médico, crédito bancario) | - Manejo de valores nulos - Codificación de variables categóricas |
| **Random Forest**        | Accuracy, RMSE, R², OOB Score                       | Ensamble de múltiples árboles de decisión que mejora la precisión y evita sobreajuste. | Número de árboles, Profundidad máxima, Número de características seleccionadas | Predicción en datos complejos, clasificación y regresión (clima, precios de mercado) | - Conversión de variables categóricas - Manejo de datos faltantes - Normalización |
| **SVM (Máquinas de Soporte Vectorial)** | Accuracy, Precision, Recall, AUC-ROC | Encuentra el hiperplano óptimo para clasificar los datos. **Fórmula:** f(x) = w * x + b | Kernel (lineal, RBF, polinomial), C, Gamma | Clasificación en datos con fronteras complejas (biometría, detección de spam) | - Estandarización de datos - Conversión de variables categóricas |
| **k-NN (k-Nearest Neighbors)** | Accuracy, Precision, Recall | Asigna una clase a una instancia según sus k vecinos más cercanos. **Distancia Euclidiana:** d(p,q) = sqrt(Σ(qi - pi)²) | Número de vecinos (k), Distancia (Euclídea, Manhattan) | Clasificación y regresión (recomendaciones, diagnóstico médico) | - Normalización/Estandarización - Conversión de variables categóricas |
| **Redes Neuronales**     | Accuracy, Loss Function, AUC-ROC                    | Modela relaciones complejas a través de capas de neuronas interconectadas. **Fórmula:** y = f(WX + B) donde f es una función de activación | Número de capas, Número de neuronas, Learning Rate, Función de activación | Reconocimiento de imágenes, procesamiento de lenguaje natural | - Normalización - Balanceo de clases - Conversión de variables categóricas - Creación de embeddings |

---

### **Métricas con sus Fórmulas:**

#### **Métricas para Regresión**
1. **Error Cuadrático Medio (MSE - Mean Squared Error)**  
   MSE = (1/n) * Σ(yi - ŷi)²  

2. **Raíz del Error Cuadrático Medio (RMSE - Root Mean Squared Error)**  
   RMSE = sqrt((1/n) * Σ(yi - ŷi)²)  

3. **Coeficiente de Determinación (R² - R-Squared)**  
   R² = 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²)  

---

#### **Métricas para Clasificación**
1. **Precisión (Precision)**  
   Precision = TP / (TP + FP)  

2. **Sensibilidad o Recall (Tasa de Verdaderos Positivos - TPR)**  
   Recall = TP / (TP + FN)  

3. **Exactitud (Accuracy)**  
   Accuracy = (TP + TN) / (TP + TN + FP + FN)  

4. **Puntuación F1 (F1-score)**  
   F1 = 2 * (Precision * Recall) / (Precision + Recall)  

5. **Área bajo la curva ROC (AUC-ROC)**  
   AUC = Integral(TPR(FPR) d(FPR))  

---

#### **Métricas para Árboles de Decisión y Random Forest**
1. **Índice de Gini (Gini Index)**  
   Gini = 1 - Σ(pi²)  

2. **Entropía (Entropy)**  
   Entropía = -Σ(pi * log2(pi))  

---

#### **Distancias en k-NN**
1. **Distancia Euclidiana**  
   d(p, q) = sqrt(Σ(qi - pi)²)  

2. **Distancia de Manhattan**  
   d(p, q) = Σ|qi - pi|  

---

Ahora puedes copiar y pegar esto directamente en un documento sin problemas. 🚀
