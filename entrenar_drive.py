import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from features import extraer_features
import random 
SEMILLA = 42
np.random.seed(SEMILLA)
random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

def cargar_dataset(carpeta, etiqueta):
    """ Carga todas las repeticiones de una carpeta y les asigna la etiqueta correspondiente (1 = correcto, 0 = incorrecto)."""
    X = []
    y = []
    for archivo in glob.glob(os.path.join(carpeta, '*.npy')):
        rep = np.load(archivo)
        features = extraer_features(rep)
        X.append(features)
        y.append(etiqueta)
    return X, y


X_correcto, y_correcto = cargar_dataset('dataset/drive/correcto', 1)
X_incorrecto, y_incorrecto = cargar_dataset('dataset/drive/incorrecto', 0)

X = np.array(X_correcto + X_incorrecto)
y = np.array(y_correcto + y_incorrecto)

print(f'Total de repeticiones: {len(X)}')
print(f'Forma de X: {X.shape}')
print(f'Correctas: {sum(y == 1)}, Incorrectas: {sum(y == 0)}')

"""Guarda el modelo entrenado en un archivo .h5 para poder usarlo luego en la app de Streamlit.
El modelo es una red neuronal simple de 3 capas densas, entrenada con las features extraidas de las repeticiones de drive
(min, max, rango y promedio de angulo de codo y rotacion de cadera)."""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)

print(f'Entrenamiento: {len(X_train)} repeticiones')
print(f'Prueba: {len(X_test)} repeticiones')

modelo = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train_escalado.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

historial = modelo.fit(
    X_train_escalado, y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

test_loss, test_acc = modelo.evaluate(X_test_escalado, y_test)
print(f'Accuracy en datos de prueba: {test_acc:.2f}')

modelo.save('modelo_drive.h5')
print('Modelo guardado en modelo_drive.h5')