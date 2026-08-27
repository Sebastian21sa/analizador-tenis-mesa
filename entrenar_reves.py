import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from features import extraer_features
import random 
SEMILLA = 42
np.random.seed(SEMILLA)
random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

def cargar_dataset(carpeta, etiqueta):
    X = []
    y = []
    for archivo in glob.glob(os.path.join(carpeta, '*.npy')):
        rep = np.load(archivo)
        features = extraer_features(rep, incluir_altura_muneca=True)
        X.append(features)
        y.append(etiqueta)
    return X, y


X_correcto, y_correcto = cargar_dataset('dataset/reves/correcto', 1)
X_incorrecto, y_incorrecto = cargar_dataset('dataset/reves/incorrecto', 0)

X = np.array(X_correcto + X_incorrecto)
y = np.array(y_correcto + y_incorrecto)

print(f'Total de repeticiones: {len(X)}')
print(f'Forma de X: {X.shape}')
print(f'Correctas: {sum(y == 1)}, Incorrectas: {sum(y == 0)}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)

print(f'Entrenamiento: {len(X_train)} repeticiones')
print(f'Prueba: {len(X_test)} repeticiones')

pesos = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: pesos[0], 1: pesos[1]}
print(f'Pesos por clase: {class_weight_dict}')

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
    #class_weight=class_weight_dict,
    verbose=1
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Accuracy durante el entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Perdida (loss) durante el entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('entrenamiento_reves.png')
print('Grafico guardado en entrenamiento_reves.png')

test_loss, test_acc = modelo.evaluate(X_test_escalado, y_test)
print(f'Accuracy en datos de prueba (reves): {test_acc:.2f}')

predicciones = modelo.predict(X_test_escalado)
predicciones_clase = (predicciones > 0.5).astype(int).flatten()

print('\n--- Detalle de cada prediccion en el set de prueba ---')
for i in range(len(y_test)):
    real = 'correcto' if y_test[i] == 1 else 'incorrecto'
    predicho = 'correcto' if predicciones_clase[i] == 1 else 'incorrecto'
    resultado = 'ACERTO' if y_test[i] == predicciones_clase[i] else 'FALLO'
    print(f'Repeticion {i+1}: real={real}, predicho={predicho} -> {resultado}')

modelo.save('modelo_reves.h5')
print('Modelo guardado en modelo_reves.h5')