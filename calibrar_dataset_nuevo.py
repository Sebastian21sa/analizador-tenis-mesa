import cv2
import numpy as np
import math
from scipy.signal import find_peaks
import mediapipe as mp

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


def calcular_velocidades(landmarks_por_frame):
    posiciones_muneca = [(f[16][0], f[16][1]) for f in landmarks_por_frame]
    velocidades = [0.0]

    for i in range(1, len(posiciones_muneca)):
        dx = posiciones_muneca[i][0] - posiciones_muneca[i - 1][0]
        dy = posiciones_muneca[i][1] - posiciones_muneca[i - 1][1]
        velocidades.append(math.sqrt(dx ** 2 + dy ** 2))

    velocidades = np.array(velocidades)
    ventana_suavizado = 5
    velocidades_suaves = np.convolve(velocidades, np.ones(ventana_suavizado) / ventana_suavizado, mode='same')

    return velocidades_suaves


def probar_umbrales(velocidades_suaves, fps, real_esperado):
    mejores = []

    for mult in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        for dist_seg in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            umbral = velocidades_suaves.mean() + mult * velocidades_suaves.std()
            distancia = max(1, int(dist_seg * fps))
            picos, _ = find_peaks(velocidades_suaves, height=umbral, distance=distancia)

            if len(picos) == real_esperado:
                mejores.append((mult, dist_seg, len(picos)))

    if mejores:
        print(f' Configs que dan EXACTAMENTE {real_esperado}:')
        for mult, dist_seg, n in mejores[:5]:
            print(f' multiplicador={mult}, distancia_seg={dist_seg}')
    else:
        print(f' Ninguna combinacion dio exacto. Mostrando las 5 mas cercanas a {real_esperado}:')
        resultados = []

        for mult in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            for dist_seg in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                umbral = velocidades_suaves.mean() + mult * velocidades_suaves.std()
                distancia = max(1, int(dist_seg * fps))
                picos, _ = find_peaks(velocidades_suaves, height=umbral, distance=distancia)
                resultados.append((abs(len(picos) - real_esperado), mult, dist_seg, len(picos)))

        resultados.sort()
        for diff, mult, dist_seg, n in resultados[:5]:
            print(f' multiplicador={mult}, distancia_seg={dist_seg} -> {n} repeticiones')


videos_a_calibrar = [
    ('raw/drive_correcto_principal.mp4', 23),
    ('raw/drive_correcto_opuesto.mp4', 30),
    ('raw/drive_correcto_frontal.mp4', 30),
    ('raw/drive_correcto_espalda.mp4', 30),
    ('raw/drive_correcto_opuesto_sombra.mp4', 16),
    ('raw/drive_correcto_frontal_sombra.mp4', 22),
    ('raw/reves_correcto_principal.mp4', 20),
    ('raw/reves_correcto_opuesto.mp4', 26),
    ('raw/reves_correcto_frontal.mp4', 26),
    ('raw/reves_correcto_espalda.mp4', 26),
    ('raw/reves_correcto_opuesto_sombra.mp4', 13),
    ('raw/reves_correcto_frontal_sombra.mp4', 19),
    ('raw/zurdo_drive_correcto_principal.mp4', 23),
    ('raw/zurdo_drive_correcto_frontal.mp4', 23),
    ('raw/zurdo_drive_correcto_espalda.mp4', 23),
    ('raw/zurdo_reves_correcto_principal.mp4', 21),
    ('raw/zurdo_reves_correcto_frontal.mp4', 21),
    ('raw/zurdo_reves_correcto_espalda.mp4', 21),
    ('raw/drive_incorrecto_opuesto_sombra.mp4', 15),
    ('raw/drive_incorrecto_frontal_sombra.mp4', 14),
    ('raw/reves_incorrecto_opuesto_sombra.mp4', 14),
    ('raw/reves_incorrecto_frontal_sombra.mp4', 12),
]

for ruta, real in videos_a_calibrar:
    print(f'\n=== {ruta} (esperado: {real}) ===')
    landmarks_por_frame, fps = extraer_landmarks_de_video(ruta)
    velocidades_suaves = calcular_velocidades(landmarks_por_frame)
    probar_umbrales(velocidades_suaves, fps, real)