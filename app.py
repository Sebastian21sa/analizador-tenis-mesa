import os 
import tempfile 
import cv2 
import numpy as np 
import joblib 
import mediapipe as mp 
from flask import Flask, request, jsonify 
from tensorflow import keras 
from features import extraer_features
"""Application Flask para analizar videos de tenis de mesa y clasificar golpes como correctos o incorrectos usando modelos entrenados."""
app = Flask(__name__) 
modelo_drive = keras.models.load_model('modelo_drive_final.h5') 
scaler_drive = joblib.load('scaler_drive.pkl') 
modelo_reves = keras.models.load_model('modelo_reves_final.h5') 
scaler_reves = joblib.load('scaler_reves.pkl')

BaseOptions = mp.tasks.BaseOptions 
PoseLandmarker = mp.tasks.vision.PoseLandmarker 
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions 
VisionRunningMode = mp.tasks.vision.RunningMode 
opciones_pose = PoseLandmarkerOptions( base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'), running_mode=VisionRunningMode.VIDEO) 

def extraer_landmarks_de_video(ruta_video):
    """ Extrae los landmarks de MediaPipe Pose de cada frame de un video."""
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
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )
            resultado = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
            if resultado.pose_landmarks:
                frame_landmarks = [
                    (lm.x, lm.y, lm.z)
                    for lm in resultado.pose_landmarks[0]
                ]
                landmarks_por_frame.append(frame_landmarks)
            frame_count += 1
    video.release()
    return landmarks_por_frame

@app.route('/predecir', methods=['POST'])
def predecir():
    """Punto final de la API que recibe un video y devuelve la predicción del golpe (correcto o incorrecto) junto con la confianza."""
    if 'video' not in request.files:
        return jsonify({'error': 'No se envio ningun video'}), 400
    golpe = request.form.get('golpe')
    if golpe not in ['drive', 'reves']:
        return jsonify({
            'error': 'El parametro golpe debe ser "drive" o "reves"'
        }), 400
    archivo_video = request.files['video']
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.mp4'
    ) as temp:
        archivo_video.save(temp.name)
        ruta_temporal = temp.name
    try:
        landmarks = extraer_landmarks_de_video(ruta_temporal)
        if len(landmarks) < 5:
            return jsonify({
                'error': 'No se detecto suficiente movimiento en el video'
            }), 400
        if golpe == 'drive':
            features = extraer_features(
                landmarks,
                incluir_altura_muneca=False
            )
            features_escalados = scaler_drive.transform([features])
            prediccion = modelo_drive.predict(
                features_escalados
            )[0][0]
        else:
            features = extraer_features(
                landmarks,
                incluir_altura_muneca=True
            )
            features_escalados = scaler_reves.transform([features])
            prediccion = modelo_reves.predict(
                features_escalados
            )[0][0]
        veredicto = 'correcto' if prediccion > 0.5 else 'incorrecto'
        confianza = (
            float(prediccion)
            if veredicto == 'correcto'
            else float(1 - prediccion)
        )
        return jsonify({
            'golpe': golpe,
            'veredicto': veredicto,
            'confianza': round(confianza, 2)
        })
    finally:
        os.remove(ruta_temporal)
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True
    )