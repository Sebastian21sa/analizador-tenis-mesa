import numpy as np 
import glob 
import os 
import random 
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf 
from tensorflow import keras 
from features import extraer_features 
SEMILLA = 42 
np.random.seed(SEMILLA) 
random.seed(SEMILLA) 
tf.random.set_seed(SEMILLA) 

def cargar_dataset(carpeta, etiqueta, incluir_altura_muneca=False):
    """ Carga un dataset de repeticiones desde una carpeta, 
    extrayendo features de cada repetición y asignando la etiqueta correspondiente.""" 
    X = [] 
    y = [] 
    for archivo in glob.glob(os.path.join(carpeta, '*.npy')): 
        rep = np.load(archivo) 
        features = extraer_features(rep, incluir_altura_muneca=incluir_altura_muneca) 
        X.append(features) 
        y.append(etiqueta) 
    return X, y

def evaluar_con_kfold(nombre_golpe, carpeta_correcto, carpeta_incorrecto, incluir_altura_muneca):
    """ Evalua un modelo de clasificacion usando validacion cruzada k-fold."""
    X_correcto, y_correcto = cargar_dataset(carpeta_correcto, 1, incluir_altura_muneca) 
    X_incorrecto, y_incorrecto = cargar_dataset(carpeta_incorrecto, 0, incluir_altura_muneca) 
    X = np.array(X_correcto + X_incorrecto) 
    y = np.array(y_correcto + y_incorrecto) 
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA) 
    accuracies = [] 
    print(f'\n=== {nombre_golpe}: {len(X)} repeticiones totales ===') 
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)): 
        X_train, X_test = X[train_idx], X[test_idx] 
        y_train, y_test = y[train_idx], y[test_idx] 
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
        _, acc = modelo.evaluate(X_test_escalado, y_test, verbose=0) 
        accuracies.append(acc) 
        print(f'Fold {fold_idx + 1}: accuracy = {acc:.2f}') 
    accuracies = np.array(accuracies) 
    print(f'Accuracy promedio ({nombre_golpe}): {accuracies.mean():.2f} +/- {accuracies.std():.2f}') 
    return accuracies

acc_drive = evaluar_con_kfold( 'DRIVE', 'dataset/drive/correcto', 'dataset/drive/incorrecto', incluir_altura_muneca=False ) 
acc_reves = evaluar_con_kfold( 'REVES', 'dataset/reves/correcto', 'dataset/reves/incorrecto', incluir_altura_muneca=True )