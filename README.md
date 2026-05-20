#  Detección de Anomalías en Series de Tiempo con LSTM Autoencoder

##  Descripción
Modelo de Deep Learning no supervisado basado en arquitectura 
LSTM Autoencoder (Encoder-Decoder) para detectar anomalías en 
series de tiempo. El modelo aprende a reconstruir patrones normales 
y detecta anomalías cuando el error de reconstrucción supera un 
umbral definido.

##  Arquitectura del Modelo
- **Encoder:** Capa LSTM que comprime secuencias de 288 pasos 
  en un vector latente de dimensión 32
- **RepeatVector:** Puente que replica el vector latente 288 veces
- **Decoder:** Capa LSTM que reconstruye la secuencia original
- **Salida:** TimeDistributed Dense para recuperar los valores reales

## Dataset
Numenta Anomaly Benchmark (NAB)
- Serie sin anomalías: `art_daily_small_noise.csv`
- Serie con anomalías: `art_daily_jumpsup.csv`

##  Resultados
- Detección exitosa de anomalías mediante umbral de error MSE
- Modelo liviano: solo 12,705 parámetros entrenables
- Arquitectura Encoder-Decoder con Dropout para evitar overfitting

##  Tecnologías utilizadas
- Python 3
- TensorFlow / Keras
- NumPy · Pandas
- Matplotlib

##  Estructura del proyecto
├── notebooks/
│   └── Reto_IV_Detección_de_anomalías_usando_LSTM-Autoencoder.ipynb
└── README.md

## 👤 Autor
**Franklin Manjarres**  
[LinkedIn](https://www.linkedin.com/in/franklinmanjarres/)
[GitHub](https://github.com/franklinmanjarres)
