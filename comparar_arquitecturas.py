import numpy as np
import glob
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from features import extraer_features

"""Script para comparar arquitecturas de redes neuronales para clasificar golpes de tenis de mesa. Carga los datos de entrenamiento, realiza validación cruzada y compara el rendimiento de diferentes arquitecturas."""
SEMILLA = 42
np.random.seed(SEMILLA)
random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)


def cargar_dataset(carpeta, etiqueta, incluir_altura_muneca=False):
    """Carga los archivos .npy de una carpeta y extrae las features de cada archivo"""
    X = []
    y = []

    for archivo in glob.glob(os.path.join(carpeta, '*.npy')):
        rep = np.load(archivo)
        features = extraer_features(rep, incluir_altura_muneca=incluir_altura_muneca)
        X.append(features)
        y.append(etiqueta)

    return X, y


def crear_modelo_pequeno(input_dim):
    modelo = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


def crear_modelo_grande(input_dim):
    """Crea un modelo más grande con capas adicionales y dropout para regularización."""
    modelo = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


def evaluar_arquitectura(nombre_arq, crear_modelo_fn, X, y):
    """Evalúa una arquitectura de red neuronal usando validación cruzada y devuelve las accuracies."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    accuracies = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_escalado = scaler.fit_transform(X_train)
        X_test_escalado = scaler.transform(X_test)

        modelo = crear_modelo_fn(X_train_escalado.shape[1])
        modelo.fit(X_train_escalado, y_train, epochs=50, verbose=0)

        _, acc = modelo.evaluate(X_test_escalado, y_test, verbose=0)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    print(f' {nombre_arq}: {accuracies.mean():.3f} +/- {accuracies.std():.3f}')
    return accuracies


def comparar_para_golpe(nombre_golpe, carpeta_correcto, carpeta_incorrecto, incluir_altura_muneca):
    X_correcto, y_correcto = cargar_dataset(carpeta_correcto, 1, incluir_altura_muneca)
    X_incorrecto, y_incorrecto = cargar_dataset(carpeta_incorrecto, 0, incluir_altura_muneca)

    X = np.array(X_correcto + X_incorrecto)
    y = np.array(y_correcto + y_incorrecto)

    print(f'\n=== {nombre_golpe}: {len(X)} repeticiones totales ===')
    evaluar_arquitectura('Red pequena (actual)', crear_modelo_pequeno, X, y)
    evaluar_arquitectura('Red grande (con dropout)', crear_modelo_grande, X, y)


comparar_para_golpe('DRIVE', 'dataset/drive/correcto', 'dataset/drive/incorrecto', incluir_altura_muneca=False)
comparar_para_golpe('REVES', 'dataset/reves/correcto', 'dataset/reves/incorrecto', incluir_altura_muneca=True)
