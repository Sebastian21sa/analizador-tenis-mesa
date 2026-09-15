import numpy as np
import glob
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from features import extraer_features


"""Script para diagnosticar el identificador de golpes de tenis de mesa. Carga los datos de entrenamiento, realiza validación cruzada y calcula los errores por grupo."""
SEMILLA = 42
np.random.seed(SEMILLA)
random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)


def cargar_carpeta(carpeta, etiqueta_golpe, nombre_grupo):
    """Carga los archivos .npy de una carpeta y extrae las features de cada archivo"""
    X = []
    y = []
    grupos = []

    for archivo in glob.glob(os.path.join(carpeta, '*.npy')):
        rep = np.load(archivo)
        features = extraer_features(rep, incluir_altura_muneca=True)
        X.append(features)
        y.append(etiqueta_golpe)
        grupos.append(nombre_grupo)

    return X, y, grupos


X1, y1, g1 = cargar_carpeta('dataset/drive/correcto', 1, 'drive_correcto')
X2, y2, g2 = cargar_carpeta('dataset/drive/incorrecto', 1, 'drive_incorrecto')
X3, y3, g3 = cargar_carpeta('dataset/reves/correcto', 0, 'reves_correcto')
X4, y4, g4 = cargar_carpeta('dataset/reves/incorrecto', 0, 'reves_incorrecto')

X = np.array(X1 + X2 + X3 + X4)
y = np.array(y1 + y2 + y3 + y4)
grupos = np.array(g1 + g2 + g3 + g4)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
errores_por_grupo = {}
total_por_grupo = {}

for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    grupos_test = grupos[test_idx]

    scaler = StandardScaler()
    X_train_escalado = scaler.fit_transform(X_train)
    X_test_escalado = scaler.transform(X_test)

    modelo = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(X_train_escalado.shape[1],)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo.fit(X_train_escalado, y_train, epochs=50, verbose=0)

    predicciones = (modelo.predict(X_test_escalado, verbose=0) > 0.5).astype(int).flatten()

    for real, pred, grupo in zip(y_test, predicciones, grupos_test):
        total_por_grupo[grupo] = total_por_grupo.get(grupo, 0) + 1
        if real != pred:
            errores_por_grupo[grupo] = errores_por_grupo.get(grupo, 0) + 1

print('Errores por grupo:')
for grupo in sorted(total_por_grupo):
    errores = errores_por_grupo.get(grupo, 0)
    total = total_por_grupo[grupo]
    print(f' {grupo}: {errores}/{total} errores ({100*errores/total:.1f}%)')
