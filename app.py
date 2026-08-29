import os 
import tempfile 
import cv2 
import numpy as np 
import joblib 
import mediapipe as mp 
from flask import Flask, request, jsonify 
from tensorflow import keras 
from features import extraer_features
import logging 
logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO) 
if not logger.handlers:
    manejador = logging.StreamHandler()
    formato = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    manejador.setFormatter(formato)
    logger.addHandler(manejador)
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

"""def detectar_repeticiones(landmarks_por_frame, fps):
    posiciones_muneca = []
    for frame in landmarks_por_frame:
        muneca = frame[16]
        posiciones_muneca.append((muneca[0], muneca[1]))
    velocidades = [0.0]
    for i in range(1, len(posiciones_muneca)):
        dx = posiciones_muneca[i][0] - posiciones_muneca[i - 1][0]
        dy = posiciones_muneca[i][1] - posiciones_muneca[i - 1][1]
        velocidades.append(math.sqrt(dx ** 2 + dy ** 2))
    velocidades = np.array(velocidades)
    ventana_suavizado = 5
    velocidades_suaves = np.convolve(
        velocidades,
        np.ones(ventana_suavizado) / ventana_suavizado,
        mode='same'
    )
    umbral_altura = velocidades_suaves.mean() + 2 * velocidades_suaves.std()
    distancia_minima = max(1, int(0.8 * fps))
    picos, _ = find_peaks(velocidades_suaves, height=umbral_altura, distance=distancia_minima)
    antes = int(0.58 * fps)
    despues = int(0.42 * fps)
    ventanas = []
    for pico in picos:
        inicio = max(0, pico - antes)
        fin = min(len(landmarks_por_frame), pico + despues)
        ventanas.append((inicio, fin))
    return ventanas"""

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
        """Extrae los landmarks del video y realiza la predicción usando el modelo correspondiente (drive o revés). Devuelve un JSON con el veredicto y la confianza de la predicción."""
        landmarks = extraer_landmarks_de_video(ruta_temporal) 
        if len(landmarks) < 5: 
            return jsonify({'error': 'No se detecto suficiente movimiento en el video'}), 400 
        if golpe == 'drive': 
            features = extraer_features(landmarks, incluir_altura_muneca=False) 
            features_escalados = scaler_drive.transform([features]) 
            prediccion = float(modelo_drive.predict(features_escalados, verbose=0)[0][0]) 
        else:
            features = extraer_features(landmarks, incluir_altura_muneca=True) 
            features_escalados = scaler_reves.transform([features]) 
            prediccion = float(modelo_reves.predict(features_escalados, verbose=0)[0][0]) 
        veredicto = 'correcto' if prediccion > 0.5 else 'incorrecto' 
        confianza = prediccion if veredicto == 'correcto' else 1 - prediccion 
        logger.info(f'Prediccion exitosa: golpe={golpe}, veredicto={veredicto}, confianza={confianza:.2f}') 
        return jsonify({ 'golpe': golpe, 'veredicto': veredicto, 'confianza': round(confianza, 2) }) 
    except Exception as e: 
        logger.error(f'Error procesando video: {str(e)}') 
        return jsonify({'error': 'Ocurrio un error procesando el video'}), 500 
    finally: 
        os.remove(ruta_temporal) 

if __name__ == '__main__': 
    puerto = int(os.environ.get('PORT', 5000)) 
    modo_debug = os.environ.get('FLASK_DEBUG', 'True') == 'True' 
app.run(host='0.0.0.0', port=puerto, debug=modo_debug)