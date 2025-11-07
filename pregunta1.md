## Adaptive Correlation-Aware Fused Lasso (ACFL)

Buscamos realizar una regularización que se comporte como L1 (Lasso) para características independientes (selección) y como L2 (Ridge) para características correlacionadas (estabilidad).

Buscamos que la propuesta no sea una simple mezcla ponderada (como Elastic Net), sino una **regularización fusionada (Fused) y adaptativa** que penaliza activamente la *diferencia* entre los coeficientes de las características correlacionadas.

Mi función de penalización propuesta, **ACFL**, es:
$$
R(w, X) = \underbrace{\lambda_1 \sum_{i=1}^p |w_i|}_{\text{Término L1 (Sparsity)}} + \underbrace{\lambda_2 \sum_{i < j} |C_{ij}| (w_i - w_j)^2}_{\text{Término Fusionado-Adaptativo (Estabilidad)}}
$$

Donde $C_{ij}$ es la correlación absoluta entre la característica $i$ y $j$.

* **¿Por qué esto funcionaría?**
* Si las características $i$ y $j$ son **independientes**, $C_{ij} \approx 0$. El segundo término desaparece y el modelo se comporta como **Lasso (L1)**, logrando la reducción de características.
* Si las características $i$ y $j$ están **altamente correlacionadas**, $C_{ij} \approx 1$. El segundo término se activa con fuerza ($\lambda_2 (w_i - w_j)^2$), forzando $w_i \approx w_j$. Esto logra el objetivo de estabilidad de **Ridge (L2)**: en lugar de elegir una característica al azar (inestabilidad de Lasso), asigna pesos similares a todo el grupo correlacionado.

Esta formulación aborda directamente el problema central y generará un rendimiento superior en los escenarios de prueba propuestos.

---

## Parte (a): Análisis Teórico (ACFL)

### 1. Propuesta de Función de Penalización $R(w, X)$

Como se mencionó, la penalización es:
$$
R(w, X) = \lambda_1 ||w||_1 + \lambda_2 \sum_{i < j} |C_{ij}| (w_i - w_j)^2
$$
La función de pérdida total $L(w)$ para el problema de regresión (usando Mínimos Cuadrados Ordinarios) es:
$$
L(w) = \frac{1}{N} ||y - Xw||_2^2 + R(w, X)
$$

### 2. Derivación del Gradiente (Subgradiente)

Para implementar esto con descenso de gradiente, debemos usar **Descenso de Gradiente Proximal (PGD)**, también conocido como ISTA, ya que el término L1 ($||w||_1$) no es diferenciable en cero.

Separamos la pérdida en una parte suave $f(w)$ y una no suave $g(w)$:
* **Parte Suave $f(w)$:** $\frac{1}{N} ||y - Xw||_2^2 + \lambda_2 \sum_{i < j} |C_{ij}| (w_i - w_j)^2$
* **Parte No Suave $g(w)$:** $\lambda_1 ||w||_1$

La actualización de PGD es: $w^{(t+1)} = \text{prox}_{\eta g} (w^{(t)} - \eta \nabla f(w^{(t)}))$

**Paso 1: Gradiente de la parte suave $\nabla f(w)$**

$\nabla f(w) = \nabla (\text{MSE}) + \nabla (\text{Término Fusionado})$

* $\nabla (\text{MSE}) = \frac{2}{N} X^T (Xw - y)$
* $\nabla (\text{Término Fusionado})$: El gradiente del término $\lambda_2 \sum_{i < j} |C_{ij}| (w_i - w_j)^2$ se puede expresar elegantemente usando el **Laplaciano de Grafo**.
    1.  Sea $A$ la matriz de adyacencia ponderada donde $A_{ij} = |C_{ij}|$ (para $i \neq j$) y $A_{ii} = 0$.
    2.  Sea $D$ la matriz diagonal de grados donde $D_{kk} = \sum_{j \neq k} A_{kj}$.
    3.  El Laplaciano es $L_C = D - A$.
    4.  El término fusionado es $\lambda_2 w^T L_C w$.
    5.  El gradiente de esta forma cuadrática es: $2 \lambda_2 L_C w$.

El gradiente total de la parte suave es:
$$
\nabla f(w) = \frac{2}{N} X^T (Xw - y) + 2 \lambda_2 L_C w
$$

**Paso 2: Operador Proximal de la parte no suave $\text{prox}_{\eta g}(z)$**

$g(w) = \lambda_1 ||w||_1$. El operador proximal para L1 es el **Operador de Umbral Blando (Soft-Thresholding Operator)**, $S$:
$$
\text{prox}_{\eta g}(z) = S_{\eta \lambda_1}(z)
$$
Donde $S_{\alpha}(z_i) = \text{sign}(z_i) \cdot \max(0, |z_i| - \alpha)$ se aplica a cada elemento.

La regla de actualización completa para nuestro algoritmo ACFL es:
$$
w^{(t+1)} = S_{\eta \lambda_1} \left( w^{(t)} - \eta \left[ \frac{2}{N} X^T (Xw^{(t)} - y) + 2 \lambda_2 L_C w^{(t)} \right] \right)
$$

### 3. Discusión de Condiciones

* **Equivalente a L1 Puro (Lasso):**
    El modelo ACFL se vuelve equivalente a Lasso si el término fusionado-adaptativo es cero. Esto ocurre bajo dos condiciones:
    1.  El hiperparámetro de estabilidad $\lambda_2 = 0$.
    2.  Todas las características son perfectamente **independientes**, lo que significa que la matriz de correlación $C$ es la identidad. En este caso, $C_{ij} = 0$ para todo $i \neq j$, el Laplaciano $L_C$ es la matriz cero, y la penalización se reduce a $\lambda_1 ||w||_1$.

* **Equivalente a L2 Puro (Ridge):**
    Es crucial notar que esta formulación **nunca** se simplifica a la forma L2 pura (Ridge, $\lambda \sum w_i^2$). Esto es una característica de diseño.
    * Ridge logra la estabilidad **encogiendo** indiscriminadamente la magnitud de *todos* los coeficientes correlacionados.
    * ACFL logra la estabilidad **forzando** que los coeficientes correlacionados tengan *valores similares* ($w_i \approx w_j$).
    * Por lo tanto, ACFL implementa el *objetivo* de L2 (estabilidad) a través de un mecanismo más inteligente y específico que es compatible con la selección de características de L1.