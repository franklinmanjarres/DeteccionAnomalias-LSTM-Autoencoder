##  Detección de Anomalías con LSTM Autoencoder


<p align="center">
  <img src="anomalias_lstm_animado.gif" width="700" alt="Detección de anomalías LSTM"/>
</p>



> Modelo de deep learning no supervisado capaz de detectar comportamientos anómalos en series temporales, sin necesidad de datos etiquetados durante el entrenamiento.

---

##  Descripción del Problema

En contextos reales —monitoreo industrial, sistemas financieros, infraestructura de servidores— detectar **cuándo algo se sale de lo normal** es crítico. El problema: casi nunca tenemos ejemplos etiquetados de las anomalías que queremos detectar.

Este proyecto propone una solución basada en **aprendizaje no supervisado**: entrenar un modelo para que aprenda el comportamiento *normal* de una serie temporal, y después detectar anomalías como aquellos puntos donde el modelo **falla en reconstruir** los datos.

---

##  Intuición del Enfoque

```
Serie normal → [LSTM Encoder] → Vector latente comprimido (dim=32)
                                        ↓
Serie reconstruida ← [LSTM Decoder] ←──┘

Error de reconstrucción bajo  → Comportamiento NORMAL
Error de reconstrucción alto  → ANOMALÍA detectada
```

El modelo actúa como un **embudo de memoria**: al obligar a los datos a pasar por una representación muy pequeña, solo retiene los patrones principales. Lo que no encaja en esos patrones genera un error de reconstrucción alto → eso es la señal de anomalía.

---

##  ¿Cómo funciona el modelo? (explicado paso a paso)

El modelo tiene una lógica muy intuitiva: **aprende qué es "normal" y detecta lo que no lo es.**

Imagina que le muestras a alguien miles de días de datos normales. Esa persona aprende a reconocer el patrón típico. Cuando le muestras un día raro, nota que algo no encaja. Eso es exactamente lo que hace este modelo.

---

### Paso 1 — Comprimir la información (Encoder)

La primera parte del modelo lee una ventana de **288 puntos de tiempo** (un día completo de datos) y los resume en un único vector pequeño de **32 números**.

$$z = \text{LSTM}_{\text{encoder}}(X) \quad \rightarrow \quad z \in \mathbb{R}^{32}$$

> **Analogía:** Es como pedirle a alguien que describa una película completa en una sola oración. Si la película es predecible y conocida, lo hará bien. Si es completamente rara y caótica, la descripción va a ser mala — y ese error es nuestra señal.

---

### Paso 2 — Preparar la reconstrucción (RepeatVector)

El resumen de 32 números no tiene dimensión temporal, pero el decodificador necesita trabajar punto por punto. Entonces simplemente **repetimos ese resumen 288 veces**, una por cada punto de tiempo que queremos reconstruir.

$$Z_{\text{seq}} = [\,z,\; z,\; \dots,\; z\,]_{\;288 \text{ veces}}$$

> **Analogía:** Es como tomar esa única oración que resume la película y dársela al modelo 288 veces para que intente reconstruir escena por escena a partir de ese resumen.

---

### Paso 3 — Reconstruir la serie original (Decoder)

La segunda parte del modelo toma ese resumen repetido e intenta **regenerar la serie original**, punto por punto.

$$\hat{H} = \text{LSTM}_{\text{decoder}}(Z_{\text{seq}})$$

---

### Paso 4 — Traducir a valores reales

Los estados internos de la red se convierten en valores numéricos concretos mediante una transformación lineal aplicada a cada instante de tiempo:

$$\hat{x}_t = W \cdot \hat{h}_t + b$$

> Donde $W$ y $b$ son los pesos que el modelo aprende durante el entrenamiento.

---

### Paso 5 — Medir el error de reconstrucción

Comparamos lo que el modelo reconstruyó contra lo que realmente ocurrió. Usamos el **Error Absoluto Medio (MAE)**, que mide en promedio cuánto se equivocó el modelo en cada punto:

$$\text{MAE} = \frac{1}{288} \sum_{t=1}^{288} \left| x_t - \hat{x}_t \right|$$

- Si el error es **bajo** → el modelo reconoció el patrón → **dato normal**  
- Si el error es **alto** → el modelo no pudo reconstruirlo → **anomalía detectada** 

---

### Paso 6 — Definir el umbral de detección

Durante el entrenamiento (solo con datos normales), calculamos el **peor error que el modelo cometió**. Ese valor se convierte en el umbral:

```
Umbral = max(MAE durante entrenamiento)

Si MAE en datos nuevos > Umbral → ANOMALÍA
```

> **¿Por qué funciona?** El modelo nunca vio anomalías mientras aprendía, entonces no sabe cómo reconstruirlas. Ese "no saber" se convierte en un error alto — y ahí está la detección.

---

### Arquitectura resumida

| Capa | Función |
|---|---|
| **LSTM Encoder** | Lee 288 pasos y los comprime en 32 números |
| **Dropout 20%** | Evita que el modelo memorice en lugar de aprender |
| **RepeatVector** | Prepara el resumen para la reconstrucción |
| **LSTM Decoder** | Reconstruye la serie punto por punto |
| **Dropout 20%** | Segunda capa de regularización |
| **Dense (salida)** | Convierte los estados internos a valores reales |

**Optimizador:** Adam · **Pérdida:** MSE · **Early Stopping:** patience=5

---

##  Dataset

**Numenta Anomaly Benchmark (NAB)** — dataset público y reproducible, sin necesidad de descarga manual.

| Archivo | Descripción |
|---|---|
| `art_daily_small_noise.csv` | Serie temporal **sin anomalías** → usada para entrenamiento |
| `art_daily_jumpsup.csv` | Serie temporal **con anomalías** (saltos abruptos) → usada para evaluación |

Los datos se descargan automáticamente desde el repositorio oficial de NAB al ejecutar el notebook.

---

## 🔁 Pipeline del Proyecto

```
1. Carga de datos (NAB)
        ↓
2. Normalización (media 0, std 1)
        ↓
3. Creación de ventanas temporales (TIME_STEPS = 288)
        ↓
4. Entrenamiento del LSTM-Autoencoder (solo datos normales)
        ↓
5. Cálculo del umbral: max(MAE) en datos de entrenamiento
        ↓
6. Evaluación en datos con anomalías
        ↓
7. Detección: muestras con MAE > umbral → ANOMALÍA
        ↓
8. Visualización de anomalías sobre la serie original
```

---

##  Resultados

- El modelo aprende a reconstruir la serie normal con error bajo y estable.
- Al evaluar sobre la serie con saltos abruptos, el error de reconstrucción **supera el umbral** exactamente en las zonas donde ocurren las anomalías.
- Las anomalías detectadas se visualizan marcadas en rojo sobre la serie temporal completa.

---

##  Tecnologías Utilizadas

- **Python 3.x**
- **TensorFlow / Keras** — construcción y entrenamiento del modelo
- **NumPy / Pandas** — manipulación de datos
- **Matplotlib** — visualización

---

##  Cómo Ejecutar

1. Abre el notebook en Google Colab o Jupyter.
2. Ejecuta todas las celdas en orden — los datos se descargan automáticamente.
3. No se requiere ninguna configuración adicional.

```bash
# Alternativamente, instala las dependencias localmente:
pip install tensorflow numpy pandas matplotlib
```

---

##  Comparación: LSTM vs CNN para Detección de Anomalías

| Característica | CNN-Autoencoder | LSTM-Autoencoder |
|---|---|---|
| Tipo de memoria | Local (ventana fija) | Secuencial (memoria del pasado) |
| Velocidad de entrenamiento | ✅ Más rápido | ⚠️ Más lento |
| Captura de dependencias largas | ❌ Limitada | ✅ Fuerte |
| Interpretabilidad | Media | Media |
| Idóneo para | Patrones locales | Series con contexto temporal largo |

**Conclusión:** Aunque el LSTM requiere más cómputo, su capacidad de "recordar" patrones pasados lo hace más robusto para series temporales con dependencias largas, donde la CNN básica tiene puntos ciegos.


