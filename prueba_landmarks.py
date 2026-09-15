import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'pose_landmarker_lite.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

video = cv2.VideoCapture('raw/prueba_lado_opuesto.mp4')

"""Extrae los landmarks de MediaPipe Pose de cada frame de un video y guarda una imagen con los landmarks detectados en el primer frame donde se detecten."""

if not video.isOpened():
    print('ERROR: no se pudo abrir el video')
else:
    print('Video abierto correctamente')

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = 0
detected_count = 0
primer_frame_guardado = False

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
            detected_count += 1
            if not primer_frame_guardado:
                for lm in resultado.pose_landmarks[0]:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.imwrite('prueba_landmarks.jpg', frame)
                primer_frame_guardado = True

        frame_count += 1

video.release()

print(f'Total de frames procesados: {frame_count}')
print(f'Frames con landmarks detectados: {detected_count}')
if frame_count > 0:
    print(f'Porcentaje de deteccion: {detected_count/frame_count*100:.1f}%')