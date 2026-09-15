import numpy as np
import glob
import os
import random
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from features import extraer_features

SEMILLA = 42
np.random.seed(SEMILLA)
random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

"""Este script entrena un modelo de red neuronal para identificar si un golpe de tenis de mesa es correcto o incorrecto, utilizando los landmarks extraídos de MediaPipe Pose."""
def cargar_carpeta(carpeta, etiqueta_golpe):
    X = []
    y = []

    for archivo in glob.glob(os.path.join(carpeta, '*.npy')):
        rep = np.load(archivo)
        features = extraer_features(rep, incluir_altura_muneca=True)
        X.append(features)
        y.append(etiqueta_golpe)

    return X, y


X1, y1 = cargar_carpeta('dataset/drive/correcto', 1)
X2, y2 = cargar_carpeta('dataset/drive/incorrecto', 1)
X3, y3 = cargar_carpeta('dataset/reves/correcto', 0)
X4, y4 = cargar_carpeta('dataset/reves/incorrecto', 0)

X = np.array(X1 + X2 + X3 + X4)
y = np.array(y1 + y2 + y3 + y4)

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

modelo = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_escalado.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.fit(X_escalado, y, epochs=50, verbose=0)

modelo.save('modelo_identificador_final.h5')
joblib.dump(scaler, 'scaler_identificador.pkl')

print(f'Modelo identificador entrenado con {len(X)} repeticiones y guardado.')
