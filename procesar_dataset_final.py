import cv2
import numpy as np
import math
import os
from scipy.signal import find_peaks
import mediapipe as mp
from normalizar_lateralidad import espejar_repeticion

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones_pose = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)


def extraer_landmarks_de_video(ruta_video):
    video = cv2.VideoCapture(ruta_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    landmarks_por_frame = []
    frame_count = 0

    with PoseLandmarker.create_from_options(opciones_pose) as landmarker:
        while video.isOpened():
            exito, frame = video.read()
            if not exito:
                break

            timestamp_ms = int(frame_count * (1000 / fps))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            if resultado.pose_landmarks:
                frame_landmarks = [(lm.x, lm.y, lm.z) for lm in resultado.pose_landmarks[0]]
                landmarks_por_frame.append(frame_landmarks)

            frame_count += 1

    video.release()
    return landmarks_por_frame, fps


def detectar_repeticiones(landmarks_por_frame, fps, multiplicador, distancia_seg):
    posiciones_muneca = [(f[16][0], f[16][1]) for f in landmarks_por_frame]
    velocidades = [0.0]

    for i in range(1, len(posiciones_muneca)):
        dx = posiciones_muneca[i][0] - posiciones_muneca[i - 1][0]
        dy = posiciones_muneca[i][1] - posiciones_muneca[i - 1][1]
        velocidades.append(math.sqrt(dx ** 2 + dy ** 2))

    velocidades = np.array(velocidades)
    ventana_suavizado = 5
    velocidades_suaves = np.convolve(velocidades, np.ones(ventana_suavizado) / ventana_suavizado, mode='same')

    umbral_altura = velocidades_suaves.mean() + multiplicador * velocidades_suaves.std()
    distancia_minima = max(1, int(distancia_seg * fps))
    picos, _ = find_peaks(velocidades_suaves, height=umbral_altura, distance=distancia_minima)

    antes = int(0.58 * fps)
    despues = int(0.42 * fps)
    ventanas = []

    for pico in picos:
        inicio = max(0, pico - antes)
        fin = min(len(landmarks_por_frame), pico + despues)
        ventanas.append((inicio, fin))

    return ventanas

# (archivo, golpe, etiqueta, es_zurdo, multiplicador, distancia_seg)
videos = [
    ('drive_correcto_principal.mp4', 'drive', 'correcto', False, 1.5, 1.0),
    ('drive_correcto_opuesto.mp4', 'drive', 'correcto', False, 1.25, 1.0),
    ('drive_correcto_frontal.mp4', 'drive', 'correcto', False, 1.75, 1.0),
    ('drive_correcto_espalda.mp4', 'drive', 'correcto', False, 1.25, 1.0),
    ('drive_correcto_opuesto_sombra.mp4', 'drive', 'correcto', False, 2.5, 0.8),
    ('drive_correcto_frontal_sombra.mp4', 'drive', 'correcto', False, 2.5, 0.8),
    ('reves_correcto_principal.mp4', 'reves', 'correcto', False, 1.25, 0.8),
    ('reves_correcto_opuesto.mp4', 'reves', 'correcto', False, 1.0, 1.0),
    ('reves_correcto_frontal.mp4', 'reves', 'correcto', False, 1.75, 0.4),
    ('reves_correcto_espalda.mp4', 'reves', 'correcto', False, 1.75, 0.8),
    ('reves_correcto_opuesto_sombra.mp4', 'reves', 'correcto', False, 1.25, 0.8),
    ('reves_correcto_frontal_sombra.mp4', 'reves', 'correcto', False, 1.5, 1.0),
    ('zurdo_drive_correcto_principal.mp4', 'drive', 'correcto', True, 2.0, 1.0),
    ('zurdo_drive_correcto_frontal.mp4', 'drive', 'correcto', True, 1.5, 0.6),
    ('zurdo_drive_correcto_espalda.mp4', 'drive', 'correcto', True, 1.5, 1.0),
    ('zurdo_reves_correcto_principal.mp4', 'reves', 'correcto', True, 1.0, 1.0),
    ('zurdo_reves_correcto_frontal.mp4', 'reves', 'correcto', True, 1.0, 1.0),
    ('zurdo_reves_correcto_espalda.mp4', 'reves', 'correcto', True, 1.75, 0.4),
    ('drive_incorrecto_opuesto_sombra.mp4', 'drive', 'incorrecto', False, 1.0, 1.0),
    ('drive_incorrecto_frontal_sombra.mp4', 'drive', 'incorrecto', False, 2.0, 0.8),
    ('reves_incorrecto_opuesto_sombra.mp4', 'reves', 'incorrecto', False, 2.0, 0.6),
    ('reves_incorrecto_frontal_sombra.mp4', 'reves', 'incorrecto', False, 2.0, 0.8),
]

CARPETA_VIDEOS = 'raw'

for archivo, golpe, etiqueta, es_zurdo, multiplicador, distancia_seg in videos:
    ruta = os.path.join(CARPETA_VIDEOS, archivo)

    if not os.path.exists(ruta):
        print(f'AVISO: no se encontro {ruta}, saltando')
        continue

    print(f'Procesando {archivo}...')
    landmarks_por_frame, fps = extraer_landmarks_de_video(ruta)

    if len(landmarks_por_frame) < 10:
        print(f' Muy pocos frames detectados, saltando')
        continue

    ventanas = detectar_repeticiones(landmarks_por_frame, fps, multiplicador, distancia_seg)
    print(f' {len(ventanas)} repeticiones detectadas')

    carpeta_destino = os.path.join('dataset', golpe, etiqueta)
    os.makedirs(carpeta_destino, exist_ok=True)
    existentes = len([f for f in os.listdir(carpeta_destino) if f.endswith('.npy')])

    for i, (inicio, fin) in enumerate(ventanas):
        repeticion = landmarks_por_frame[inicio:fin]

        if es_zurdo:
            repeticion = espejar_repeticion(repeticion)

        numero = existentes + i + 1
        nombre_salida = os.path.join(carpeta_destino, f'{numero:03d}.npy')
        np.save(nombre_salida, np.array(repeticion))

    print(f' Guardado en {carpeta_destino}')

print('Listo, dataset ampliado procesado.')
