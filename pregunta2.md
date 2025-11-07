# Constrained-Capacity K-Means(CCKM)

## Concepto General

El K-Means estándar falla porque su paso de asignación es "ingenuo": cada punto simplemente va a su centroide más cercano, ignorando cualquier regla.

Nuestra modificación, CCKM, introduce inteligencia en tres etapas:
1.  **Pre-procesamiento:** Se manejan las restricciones **Must-Link (ML)** de forma definitiva fusionando puntos en "super-nodos". Esto *garantiza* que nunca se separen.
2.  **Paso de Asignación Modificado:** Se modifica para que una asignación solo sea "válida" si respeta las restricciones **Cannot-Link (CL)** y de **Capacidad Máxima**.
3.  **Post-procesamiento:** Se añade un paso heurístico para "reparar" los clusters que no cumplan con la **Capacidad Mínima**, ya que K-Means no optimiza para esto de forma nativa.

---

### 1. Pre-procesamiento (Manejo de Restricciones Estáticas)

Antes de que comience K-Means, "cocinamos" las restricciones que podamos.

1.  **Fusionar Must-Links:**
    * Construimos un grafo donde cada punto de datos es un nodo. Cada restricción `Must-Link(A, B)` es una arista.
    * Encontramos todas las componentes conexas de este grafo (usando un algoritmo como *Union-Find* o Búsqueda en Profundidad).
    * Cada componente conexa se convierte en un **"super-nodo"** (o meta-punto).
    * A partir de ahora, el algoritmo no clusterizará `N` puntos, sino `M` super-nodos (donde `M <= N`).
    * Cada super-nodo `s` tiene dos propiedades:
        * **Tamaño `|s|`**: El número de puntos originales que contiene.
        * **Centroide `c(s)`**: La media de las coordenadas de todos los puntos originales que contiene.

2.  **Validar Contradicciones:**
    * Revisamos la lista de restricciones **Cannot-Link (CL)**.
    * Para cada `Cannot-Link(A, B)`:
        * Verificamos si `A` y `B` pertenecen al *mismo* super-nodo (debido a una cadena de Must-Links).
        * Si es así, las restricciones son **infactibles**. El algoritmo debe detenerse y reportar la contradicción (ej: "Se pide que A y B estén juntos, pero también que estén separados").
    * Almacenamos las restricciones CL en una estructura eficiente (como un `set` de tuplas) usando los IDs de los *super-nodos*.

---

### 2. Algoritmo: Constrained-Capacity K-Means (CCKM)

Una vez pre-procesado, el algoritmo sigue una estructura K-Means, pero con pasos modificados.

1.  **Inicialización:**
    * Inicializar `K` centroides de cluster `C_1, ..., C_K`.
    * *Recomendación:* Usar K-Means++ sobre los *centroides* de los `M` super-nodos.

2.  **Iteración (hasta convergencia):**

    * **Paso de Asignación (Modificado):**
        * Este es el paso más crítico. No podemos simplemente asignar cada super-nodo a su centroide más cercano.
        * Usamos una **asignación iterativa de re-equilibrio**:
        * Mantenemos las asignaciones de la iteración anterior.
        * Iteramos (múltiples veces o hasta que no haya cambios) sobre todos los super-nodos `s_i` en orden aleatorio.
        * Para cada `s_i`:
            1.  Calculamos la "distancia" (costo) a cada centroide de cluster `C_j`.
            2.  Buscamos el **"mejor cluster válido"** `C_j` para `s_i`.
            3.  Un cluster `C_j` es **"válido"** si:
                * **Validez CL:** `s_i` NO tiene una restricción CL con *ningún* otro super-nodo `s_k` ya asignado a `C_j`.
                * **Validez MaxCap:** `TamañoActual(C_j) + |s_i| <= MaxCapacity` (a menos que `s_i` ya esté en `C_j`).
            4.  El "mejor" es el cluster *válido* con la menor distancia (costo).
            5.  Asignamos `s_i` a este "mejor cluster válido".
        * *Conflicto:* Si un super-nodo `s_i` no tiene *ningún* cluster válido, se marca como "no asignado" o se deja en su cluster anterior.

    * **Paso de Actualización (Modificado):**
        * Para cada cluster `C_j`:
        * El nuevo centroide se calcula como la **media ponderada** de los centroides de los super-nodos asignados a él.
        * $$C_j = \frac{\sum_{s_i \in \text{Cluster}_j} |s_i| \cdot c(s_i)}{\sum_{s_i \in \text{Cluster}_j} |s_i|}$$
        * (Esto es matemáticamente equivalente a la media de todos los *puntos originales* en ese cluster).

3.  **Post-procesamiento (Manejo de Capacidad Mínima):**
    * El algoritmo anterior no garantiza la capacidad mínima. Añadimos una heurística de "reparación":
    * Identificar clusters "deficientes" (`|C_j| < MinCapacity`) y clusters "donantes" (`|C_j| > MinCapacity`).
    * Mientras existan clusters deficientes:
        1.  Tomar un cluster deficiente `C_u`.
        2.  Buscar en los clusters donantes `C_d` el "mejor super-nodo para mover" `s_d`.
        3.  El "mejor" `s_d` es uno que:
            * Está "cerca" del centroide de `C_u`.
            * **Movimiento Válido:** El movimiento `s_d \rightarrow C_u` debe ser válido (no viola CL y no viola MaxCapacity en `C_u`).
        4.  Mover el mejor super-nodo encontrado. Si no se encuentra ninguno, el problema puede ser infactible con esta heurística.

### 3. Pseudocódigo

```plaintext
Función CCKM(Puntos, ML, CL, K, MaxCap, MinCap):
    SuperNodos = Fusionar_Must_Links(Puntos, ML)
    Validar_Restricciones(SuperNodos, CL)
    Centroides = Inicializar_Centroides(SuperNodos, K)
    Asignaciones = {}
    
    Repretir hasta convergencia:
        TamañoClusters = Calcular_Tamaños(Asignaciones, SuperNodos)
        Cambios = True
        Mientras Cambios:
            Cambios = False
            Para cada s_i en SuperNodos (en orden aleatorio):
                ClusterActual = Asignaciones.get(s_i)
                MejorCluster = ClusterActual
                MinCosto = Infinito

                Si ClusterActual no es Nulo:
                    MinCosto = Distancia(s_i.centroide, Centroides[ClusterActual])
                    TamañoClusters[ClusterActual] -= s_i. tamaño

                Para j desde 1 hasta K:
                    Costo = Distancia(s_i.centroide, Centroides[j])

                    Si Costo < MinCosto:
                        Es_Valido_CL = True
                        Para s_k en Cluster[j]:
                            Si Hay_CL(s_i, s_k):
                                Es_Valido_CL = False
                                break
                        
                        Es_Valido_MaxCap = (TamañoClusters[j] + s_i.tamaño <= MaxCap)

                        Si Es_Valido_CL y Es_Valido_MaxCap:
                            MiCosto = Costo
                            MejorCluster = j
                
                Si MejorCluster != ClusterActual:
                    Asignaciones[s_i] = MejorCluster
                    Cambios = True

                TamañosCluster[MejorCluster] += s_i.tamaño

        NuevosCentroides = Calcular_Centroides_Ponderados(SuperNodos, Asignaciones)

        Si NuevosCentroides == Centroides:
            break
        Centroides = NuevosCentroides

```