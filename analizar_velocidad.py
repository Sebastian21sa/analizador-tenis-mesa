import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.signal import find_peaks

# ============ CAMBIA ESTO PARA CADA VIDEO ============
VIDEO_PATH = 'raw/reves_incorrecto_raw.mp4'
CARPETA_SALIDA = 'dataset/reves/incorrecto'
HEIGHT = 0.008
DISTANCE = 65
# =======================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)

posiciones_muneca = []
todos_los_landmarks = []

frame_count = 0
with PoseLandmarker.create_from_options(options) as landmarker:
    while video.isOpened():
        exito, frame = video.read()
        if not exito:
            break

        timestamp_ms = int(frame_count * (1000 / fps))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

        if resultado.pose_landmarks:
            muneca = resultado.pose_landmarks[0][16]
            posiciones_muneca.append((muneca.x, muneca.y))
            frame_landmarks = [(lm.x, lm.y, lm.z) for lm in resultado.pose_landmarks[0]]
            todos_los_landmarks.append(frame_landmarks)
        else:
            posiciones_muneca.append(None)
            todos_los_landmarks.append(None)

        frame_count += 1

video.release()
print(f'Total de frames con posicion de muneca: {len(posiciones_muneca)}')

velocidades = []
for i in range(1, len(posiciones_muneca)):
    p_anterior = posiciones_muneca[i - 1]
    p_actual = posiciones_muneca[i]
    if p_anterior is None or p_actual is None:
        velocidades.append(0)
        continue
    dx = p_actual[0] - p_anterior[0]
    dy = p_actual[1] - p_anterior[1]
    distancia = math.sqrt(dx**2 + dy**2)
    velocidades.append(distancia)

print(f'Velocidad minima: {min(velocidades):.4f}')
print(f'Velocidad maxima: {max(velocidades):.4f}')
print(f'Velocidad promedio: {sum(velocidades)/len(velocidades):.4f}')


def suavizar(datos, ventana=5):
    suavizado = []
    for i in range(len(datos)):
        inicio = max(0, i - ventana // 2)
        fin = min(len(datos), i + ventana // 2 + 1)
        suavizado.append(sum(datos[inicio:fin]) / (fin - inicio))
    return suavizado


velocidades_suaves = suavizar(velocidades)

plt.figure(figsize=(14, 5))
plt.plot(velocidades_suaves)
plt.title('Velocidad de la muneca a lo largo del video')
plt.xlabel('Frame')
plt.ylabel('Velocidad (suavizada)')
plt.savefig('velocidad_muneca.png')
print('Grafico guardado en velocidad_muneca.png')

picos, _ = find_peaks(velocidades_suaves, height=HEIGHT, distance=DISTANCE)
print(f'Repeticiones detectadas: {len(picos)}')
print(f'Frames donde ocurren: {list(picos)}')

ANTES = 35
DESPUES = 25

for idx, pico in enumerate(picos):
    inicio = max(0, pico - ANTES)
    fin = min(len(todos_los_landmarks), pico + DESPUES)
    ventana = todos_los_landmarks[inicio:fin]

    if any(f is None for f in ventana):
        print(f'Repeticion {idx+1}: se salto, tuvo frames sin deteccion')
        continue

    array_repeticion = np.array(ventana)
    ruta_salida = os.path.join(CARPETA_SALIDA, f'rep_{idx+1:02d}.npy')
    np.save(ruta_salida, array_repeticion)

print(f'Guardadas {len(picos)} repeticiones en {CARPETA_SALIDA}')