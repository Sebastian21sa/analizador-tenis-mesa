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

def entrenar_modelo_final(nombre_golpe, carpeta_correcto, carpeta_incorrecto, incluir_altura_muneca): 
    X_correcto, y_correcto = cargar_dataset(carpeta_correcto, 1, incluir_altura_muneca) 
    X_incorrecto, y_incorrecto = cargar_dataset(carpeta_incorrecto, 0, incluir_altura_muneca) 
    X = np.array(X_correcto + X_incorrecto) 
    y = np.array(y_correcto + y_incorrecto) 
    scaler = StandardScaler() 
    X_escalado = scaler.fit_transform(X) 
    modelo = keras.Sequential([ 
        keras.layers.Dense(16, activation='relu', input_shape=(X_escalado.shape[1],)), 
        keras.layers.Dense(8, activation='relu'), 
        keras.layers.Dense(1, activation='sigmoid') 
    ]) 
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    modelo.fit(X_escalado, y, epochs=50, verbose=0) 
    modelo.save(f'modelo_{nombre_golpe}_final.h5') 
    joblib.dump(scaler, f'scaler_{nombre_golpe}.pkl') 
    print(f'Modelo final de {nombre_golpe} entrenado con {len(X)} repeticiones y guardado.') 

entrenar_modelo_final('drive', 'dataset/drive/correcto', 'dataset/drive/incorrecto', incluir_altura_muneca=False) 
entrenar_modelo_final('reves', 'dataset/reves/correcto', 'dataset/reves/incorrecto', incluir_altura_muneca=True) 