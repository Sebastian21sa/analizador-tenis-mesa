# Documentación Técnica — Analizador de Técnica de Tenis de Mesa

Este es un documento de diseño del proyecto: un sistema de visión por computador que analiza la técnica del drive y el revés en tenis de mesa, usando MediaPipe Pose y un modelo de deep learning entrenado con un dataset propio.

---

## 1. Criterios técnicos de los golpes

Los criterios son definidos a partir de mi propia experiencia como jugador, formalizados para su uso con **MediaPipe Pose**.

### 1.1 Landmarks utilizados

MediaPipe Pose devuelve 33 puntos del cuerpo. Los relevantes para este proyecto:

| Punto del cuerpo | Índice — lado derecho | Índice — lado izquierdo |
|---|---|---|
| Hombro | 12 | 11 |
| Codo | 14 | 13 |
| Muñeca | 16 | 15 |
| Cadera | 24 | 23 |
| Rodilla | 26 | 25 |
| Tobillo | 28 | 27 |
| Nariz (referencia de cabeza) | 0 | 0 |

### 1.2 Ángulos y medidas derivadas

- **Ángulo de codo**: hombro–codo–muñeca (brazo dominante).
- **Rotación de tronco/cadera**: ángulo entre la línea de caderas y la línea de hombros, comparado contra la posición base.
- **Transferencia de peso**: desplazamiento lateral del punto medio entre tobillos, o de la cadera dominante, entre fases.
- **Altura relativa de muñeca**: posición vertical de la muñeca respecto al hombro (clave para el revés).
- **Estabilidad del codo**: variación de posición del codo entre la fase de preparación y la de impacto.

> **Limitación conocida:** la rotación de muñeca/antebrazo (orientación de la palma) no es medible de forma confiable solo con MediaPipe Pose, ya que el modelo entrega posición de puntos, no orientación. Una mejora futura: incorporar MediaPipe Hands.


### 1.3 Posición base (stance neutral)

- Piernas abiertas al ancho de hombros o más.
- Pie contrario a la mano dominante levemente adelantado.
- Ángulo de codo ≈ 90°.
- Cabeza orientada al frente.

### 1.4 Drive

| Fase | Descripción | Landmarks / ángulos clave |
|---|---|---|
| Posición base | Ángulo de codo ≈ 90°, stance neutral | Ángulo codo, cadera neutral |
| Armado (backswing) | Rotación de tronco hacia el lado dominante; transferencia de peso | Rotación cadera-hombro, desplazamiento lateral |
| Impacto | Retorno de cadera y peso; antebrazo avanza al frente; **el codo no se desplaza** | Ángulo codo estable, rotación invertida, desplazamiento de peso |
| Retorno | Vuelve a posición base | — |

**Errores comunes (incorrecto):** desplazamiento del codo, movimiento aislado de muñeca, ausencia de transferencia de peso, golpe ejecutado solo con el brazo, rango de movimiento exagerado.

### 1.6 Revés

| Fase | Descripción | Landmarks / ángulos clave |
|---|---|---|
| Posición base | Igual que el drive | Ángulo codo, stance neutral |
| Armado | Antebrazo hacia el cuerpo, altura de ombligo | Posición de muñeca (altura) |
| Impacto | Antebrazo hacia altura del hombro dominante; peso proyectado adelante | Posición de muñeca vs. hombro, desplazamiento de peso |
| Retorno | Vuelve a posición base | — |

**Errores comunes:** los mismos generales del drive (codo, brazo aislado, falta de peso, rango exagerado).

---

## 2. Arquitectura del sistema

Dos flujos separados: entrenamiento (offline) e inferencia (producción).

**Pipeline de entrenamiento:**
Dataset de video → MediaPipe Pose (landmarks) → Preprocesamiento (normalización + ángulos) → Entrenamiento (TensorFlow/Keras) → Modelo guardado

**Pipeline de inferencia:**
Video del usuario → API Flask (Docker) → Pose + preprocesamiento (mismo pipeline) → Modelo cargado → Respuesta JSON → Frontend muestra resultado

**Capas transversales:** CI/CD (GitHub Actions) corriendo tests en cada push; despliegue con Docker (backend) y build estático (frontend).

---

## 3. Dataset

Segmentación automática por velocidad de movimiento de la muñeca (`scipy.signal.find_peaks`), calibrada por golpe:

| Categoría | Umbral (height, distance) | Repeticiones |
|---|---|---|
| Drive correcto | 0.018, 80 | 23 |
| Drive incorrecto | 0.018, 80 | 16 |
| Revés correcto | 0.008, 65 | 20 |
| Revés incorrecto | 0.008, 65 | 12 |
| **Total** | | **71 repeticiones** |

Existe un leve desbalance de clases (revés incorrecto es la categoría más pequeña).

---

## 4. Modelado

### 4.1 Arquitectura del modelo

Red neuronal densamente conectada (Dense / Multi-Layer Perceptron), no recurrente ni convolucional:

```
Entrada (8 o 12 features) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

**Justificación:** cada repetición se resume en un vector fijo de features (min/max/rango/promedio de ángulos), en vez de usar la secuencia completa de frames. Con un dataset pequeño (32-39 ejemplos por golpe), una red recurrente tendría alto riesgo de sobreajuste — una red densa sobre features resumidos es la elección apropiada para datos tabulares de tamaño reducido.

- Optimizador: Adam. Función de pérdida: binary crossentropy.

### 4.2 Resultados

Validados con **5-fold cross-validation** (más confiable que un unico entrenamiento, dado el tamaño del dataset):

| Golpe | Accuracy promedio |
|---|---|
| Drive | 74% ± 13% |
| Revés | 76% ± 15% |

Ambos golpes se mantienen consistentemente por encima del 50% (azar en clasificación binaria), confirmando que el modelo captura una señal real. La desviación estándar amplia refleja honestamente el tamaño reducido del dataset.

### 4.3 Modelos de producción

Los modelos servidos por la API se entrenan con el 100% del dataset (no con partición 80/20), ya que la validación cruzada confirmó que el enfoque generaliza. Se guardan junto con su `StandardScaler` correspondiente (vía `joblib`), necesario porque el scaler aprende parámetros específicos del dataset de entrenamiento que deben aplicarse igual a cualquier dato nuevo.

---

## 5. API (Backend)

Endpoint `POST /predecir`: recibe un video y el parámetro `golpe` (`drive` o `reves`).

```
Video → archivo temporal → MediaPipe Pose → extraer_features()
→ scaler.transform() → modelo.predict() → JSON de respuesta
→ limpieza del archivo temporal (try/finally)
```

**Diseño de alcance — un video, un golpe:** se evaluó una versión que aceptaba videos largos con múltiples repeticiones, detectándolas automáticamente con un umbral adaptativo (`media + 2·desviación estándar` de la señal de velocidad). Resultado de prueba: 26 repeticiones detectadas vs. 23 reales (~13% de sobreconteo). Se decidió revertir a la versión simple: el objetivo del producto es la precisión del diagnóstico técnico, no el conteo de repeticiones — introducir una fuente de error adicional (segmentación) sobre la ya existente (accuracy del modelo) no aportaba valor al caso de uso real.

**Manejo de errores:** excepciones inesperadas se registran en el log con detalle completo; al usuario se le devuelve un mensaje genérico (HTTP 500) sin exponer información interna del sistema.

**Configuración:** puerto y modo debug leídos de variables de entorno (`PORT`, `FLASK_DEBUG`), necesario para despliegue en servicios que asignan el puerto dinámicamente.

---

## 6. Frontend

Se utilizo el framework React con Vite. donde: golpe seleccionado, pestaña activa, resultado, estado de carga.

**Dos formas de entrada, un mismo flujo:** subir archivo de video o grabar directo con la cámara del navegador. Ambas comparten una única función de envío (`analizarVideo`) hacia el mismo endpoint `/predecir`.

**CORS:** El backend y el frontend corren en puertos distintos; se habilitó `flask-cors` para permitir la comunicación entre ambos, restricción estándar de seguridad de los navegadores.

**Identidad visual:** diseño propio inspirado en marcadores deportivos.

---

## 7. Calidad de código

**Refactor:** la lógica de cálculo de ángulos y extracción de features, antes duplicada entre scripts de entrenamiento, se centralizó en un módulo compartido (`features.py`) con un parámetro que activa el feature adicional del revés.

**Tests unitarios (pytest):** 5 tests cubriendo la lógica más crítica — cálculo de ángulos contra casos geométricos conocidos, y extracción de features contra casos sintéticos controlados.

**CI/CD (GitHub Actions):** Se hizo un pipeline que instala únicamente las dependencias que los tests requieren (`pytest`, `numpy` — no TensorFlow/MediaPipe/OpenCV, que los tests no usan), y corre la suite en cada `push` o pull request.

---

## 8. Lecciones de ingeniería

Problemas reales encontrados y resueltos durante el desarrollo:

- **Reproducibilidad:** sin fijar una semilla aleatoria, el mismo código podía dar accuracy muy distinto entre corridas (86% → 57%) por la inicialización aleatoria de pesos. Solución: fijar semillas en NumPy, `random` y TensorFlow.
- **`class_weight` contraproducente:** con un dataset pequeño y desbalanceado, balancear las clases penalizando más la minoritaria causó que el modelo colapsara y predijera siempre la misma clase — un caso donde una técnica válida en general no era la correcta para este contexto específico. Se validó empíricamente, no se asumió.
- **Logging silencioso:** `logging.basicConfig()` no tiene efecto si el logger raíz ya tiene handlers configurados (Flask/Werkzeug los configura antes). Solución: configurar el logger de la aplicación directamente.

---

## 9. Limitaciones conocidas y trabajo futuro

- **Dataset pequeño** (71 repeticiones): suficiente para validar el enfoque, insuficiente para uso en producción real.
- **Sensibilidad fuera de distribución (OOD):** el modelo asume que el usuario selecciona correctamente el tipo de golpe. Ante una entrada de un golpe distinto al seleccionado, puede predecir con alta confianza de forma incorrecta — limitación conocida de clasificadores binarios sin clase de "rechazo".
- **Sensibilidad al ángulo de cámara:** el dataset se grabó desde un único ángulo lateral; videos desde ángulos muy distintos pueden degradar la precisión.

**Plan de mejora:**
1. Ampliar el dataset grabando desde 2-3 ángulos de cámara distintos.
2. Rediseñar el pipeline en dos etapas: (1) identificar qué golpe es (drive, revés, o no reconocido), (2) evaluar si fue correcto — resolviendo el problema de OOD desde la raíz en vez de depender de la selección manual del usuario.
3. Explorar el uso de la coordenada `z` (profundidad) que MediaPipe Pose ya entrega, como ayuda para normalizar variaciones de ángulo y distancia.

---

## 10. Despliegue en producción

**Backend (Render):** contenedorizado con Docker. Decisión clave: el `Dockerfile` reentrena el modelo automáticamente en cada build (`RUN python entrenar_final.py`), en vez de subir los archivos `.h5`/`.pkl` ya entrenados al repositorio. Justificación: evita que el modelo desplegado quede desincronizado del dataset si en el futuro se reentrenaba y se olvidaba regenerar/subir los archivos manualmente — dado que las semillas aleatorias ya estaban fijadas (ver sección 8), este reentrenamiento en build es completamente reproducible.

**Problemas reales resueltos durante el despliegue:**
- **Librerías gráficas faltantes:** la imagen base `python:3.13-slim` no incluye las librerías de sistema que MediaPipe necesita para renderizado (`libEGL.so.1` y otras). Se identificaron iterativamente probando el contenedor localmente antes de desplegar, y se agregaron vía `apt-get install` en el Dockerfile (`libgl1`, `libglib2.0-0`, `libegl1`, `libgles2`, `libsm6`, `libxext6`).
- **Límite práctico de tamaño de petición:** al probar la API en producción con un video de prueba de ~250 MB (una sesión completa de práctica, no un solo golpe), el servidor respondió `502 Bad Gateway` — el plan gratuito de Render no sostiene peticiones tan pesadas ni el tiempo de procesamiento que implicarían con los recursos de CPU limitados del nivel gratuito. Al probar con un clip de un solo golpe (unos pocos MB, el caso de uso real de la API), la petición se resolvió sin problema. Este hallazgo confirma y refuerza la decisión de diseño de "un video, un golpe" tomada en la sección 5.

**Frontend (Vercel):** desplegado desde la subcarpeta `frontend/` del monorepo (Root Directory configurado explícitamente). La URL del backend ya no está fija en el código — se inyecta vía variable de entorno (`VITE_API_URL`), distinta en desarrollo local (`.env`, apuntando a `127.0.0.1:5000`) y en producción (configurada en el panel de Vercel, apuntando a la URL de Render).

**Resultado:** proyecto completo accesible públicamente de punta a punta — frontend en Vercel comunicándose con backend en Render, sin dependencia de que ninguna máquina personal esté encendida.

**Demo en vivo:** [pongiq-murex.vercel.app](https://pongiq-murex.vercel.app)
