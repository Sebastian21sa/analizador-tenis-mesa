import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones_pose = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE)

RUTA_VIDEO = r"D:\Dataset Poniq\drive_correcto_frontal.mp4"
FRAME_A_CAPTURAR = 80
SALIDA = 'landmarks_frontal.png'

video = cv2.VideoCapture(RUTA_VIDEO)
video.set(cv2.CAP_PROP_POS_FRAMES, FRAME_A_CAPTURAR)
exito, frame = video.read()
video.release()

if exito:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    with PoseLandmarker.create_from_options(opciones_pose) as landmarker:
        resultado = landmarker.detect(mp_image)

    if resultado.pose_landmarks:
        alto, ancho = frame.shape[:2]
        landmarks = resultado.pose_landmarks[0]

        conexiones = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

        for inicio, fin in conexiones:
            x1, y1 = int(landmarks[inicio].x * ancho), int(landmarks[inicio].y * alto)
            x2, y2 = int(landmarks[fin].x * ancho), int(landmarks[fin].y * alto)
            cv2.line(frame, (x1, y1), (x2, y2), (232, 179, 61), 3)

        for lm in landmarks:
            x, y = int(lm.x * ancho), int(lm.y * alto)
            cv2.circle(frame, (x, y), 6, (127, 203, 232), -1)

        cv2.imwrite(SALIDA, frame)
        print(f'Guardado en {SALIDA}')
    else:
        print('No se detecto pose en ese frame, prueba con otro FRAME_A_CAPTURAR')
