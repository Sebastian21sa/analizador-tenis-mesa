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

Segmentación automática por velocidad de movimiento de la muñeca (`scipy.signal.find_peaks`).

**Estado actual (tras la ampliación documentada en la sección 14):**

| Categoría | Repeticiones |
|---|---|
| Drive correcto | 243 |
| Drive incorrecto | 45 |
| Revés correcto | 212 |
| Revés incorrecto | 38 |
| **Total** | **538 repeticiones** |

Cubre 4 ángulos de cámara (lateral original, lateral opuesto, frontal, espalda) y ambas manos (diestro y zurdo, este último normalizado con espejo de lateralidad — ver sección 14).

**Desbalance de clases:** dentro de cada golpe, "correcto" supera ampliamente a "incorrecto" (243 vs. 45 en drive, 212 vs. 38 en revés) — esperable, ya que se grabaron más ángulos y repeticiones extra de "correcto" (incluyendo variantes de sombra) que de "incorrecto". Entre golpes, el total está balanceado (288 drive vs. 250 revés).

*Nota histórica: la tabla original de calibración manual por video (umbrales `height`/`distance` específicos de las primeras 71 repeticiones) se conserva en el historial de commits del proyecto; el dataset ampliado se procesó con calibración dirigida por video, ver sección 14.*

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

---

## 11. Monitoreo y experiencia de uso

**Endpoint `/health`:** ruta GET que confirma que el servicio está activo y que ambos modelos (drive y revés) se cargaron correctamente al arrancar, sin necesidad de procesar un video completo. Punto de entrada estándar para monitoreo automatizado.

**Validación proactiva de tamaño de video:** a raíz del `502 Bad Gateway` descubierto al desplegar (sección 10), se agregó una validación en el frontend que rechaza archivos mayores a 20 MB *antes* de enviarlos, con un mensaje explicando el porqué — en vez de que el usuario espere y se tope con un error del servidor sin contexto.

**Aviso de cold start:** dado que el plan gratuito de Render duerme el servicio tras inactividad, se agregó una nota visible durante el estado de carga advirtiendo que la primera petición puede tardar hasta un minuto — transparencia sobre una limitación real del hosting gratuito, en vez de dejar que se sienta como que la aplicación está "colgada".

---

## 12. Fase de mejora — Dataset multi-ángulo

Siguiendo el plan de mejora de la sección 9 (resolver la sensibilidad al ángulo de cámara), se amplió el alcance del dataset a **4 ángulos de cámara**: lateral original, lateral opuesto, frontal, y espalda — más una variante con jugador zurdo (pendiente de coordinar).

**Validación previa a grabar:** antes de invertir tiempo grabando el dataset completo en cada ángulo nuevo, se probó la detección de MediaPipe Pose con clips cortos de prueba en los 3 ángulos nuevos. Los 3 dieron 100% de detección de frames (el modelo encontró un cuerpo en cada frame).

**Hallazgo técnico importante: detección no es lo mismo que precisión.** Al inspeccionar visualmente el ángulo "lateral opuesto", se notó que algunos landmarks no coincidían con la posición real del cuerpo, a pesar del 100% de detección reportado. Explicación: el campo `pose_landmarks` solo indica si MediaPipe encontró *algún* cuerpo en el frame — no garantiza que cada punto individual esté correctamente ubicado. Cuando una articulación queda ocluida (el brazo dominante pasando detrás del torso, en este ángulo específico), MediaPipe no reporta "no visible" — infiere una posición plausible basada en su entrenamiento previo, y esa posición puede no coincidir con la realidad, aunque cuente como "detectado".

**Solución planeada:** cada landmark de MediaPipe Pose incluye un campo `visibility` (0 a 1) que sí refleja la confianza de que el punto es realmente visible (no inferido). En la etapa de procesamiento del dataset ampliado, se filtrarán o se les restará peso a los frames donde la visibilidad del codo/muñeca del lado dominante sea baja, en vez de calcular features sobre posiciones que el modelo está adivinando.

**Decisión de alcance:** se optó por conservar los 4 ángulos (incluyendo el "lado opuesto" con la limitación conocida) en vez de descartarlo, priorizando que el sistema funcione razonablemente sin importar cómo un usuario real decida grabar — con la mitigación de visibilidad aplicada en el preprocesamiento.

---

## 13. Keep-alive de la API (GitHub Actions)

**Problema:** el plan gratuito de Render duerme el servicio tras 15 minutos de inactividad, generando una demora de 30-60s en la primera petición de cualquier usuario nuevo.

**Solución:** un segundo workflow de GitHub Actions (`.github/workflows/keep-alive.yml`), independiente del de tests, con un disparador `schedule` (cron `*/10 * * * *` — cada 10 minutos) que hace `curl` al endpoint `/health`. Al mantener peticiones constantes por debajo del umbral de 15 minutos de inactividad, el servicio se mantiene activo sin depender de tráfico real de usuarios.

**Limitación honesta documentada:** GitHub no garantiza precisión exacta en la ejecución de workflows programados — en momentos de alta demanda en su infraestructura, puede haber demoras de varios minutos respecto al intervalo configurado. Es la mejor solución gratuita disponible, no una garantía absoluta de que el servicio nunca se duerma.

---

## 14. Ampliación del dataset — 4 ángulos, zurdo, y práctica tipo "sombra"

**Grabación inicial (primera sesión):** 16 videos, cubriendo drive y revés, diestro y zurdo, en 4 ángulos de cámara (lateral original, lateral opuesto, frontal, espalda) — todos etiquetados "correcto". Los "incorrecto" originales del dataset base solo existen para el ángulo lateral original, por lo que las categorías nuevas (3 ángulos adicionales + zurdo) quedan incompletas hasta grabar sus propios incorrectos.

**Decisión: grabar los incorrectos faltantes como "sombra"** (el movimiento del golpe exagerando los errores técnicos conocidos, sin pelota ni mesa reales), en vez de con pelota real.

**Justificación técnica (no es solo conveniencia):** el pipeline de segmentación y extracción de features se basa únicamente en el movimiento del cuerpo del jugador (velocidad de muñeca, ángulos articulares) — MediaPipe Pose nunca rastrea la pelota, así que una repetición de sombra genera exactamente el mismo tipo de señal útil que una repetición real.

**Justificación de producto (la más importante):** un usuario real de la aplicación no siempre va a tener acceso a una mesa y pelota al momento de grabarse — es razonable esperar que muchos usuarios graben movimientos tipo sombra. Si el dataset de entrenamiento solo contuviera repeticiones con pelota/mesa real, un usuario grabando sombra sería otro caso de entrada fuera de distribución (mismo problema documentado en la sección 10). Incluir sombra en el entrenamiento desde el inicio alinea mejor la distribución de entrenamiento con el uso real esperado de la aplicación, en vez de ser solo una forma más rápida de generar volumen de datos.

**Matiz técnico a vigilar:** una repetición de sombra podría tener un ritmo ligeramente distinto a una repetición real (en la real, el cuerpo reacciona al vuelo de la pelota; en la sombra, el jugador controla el tiempo completo). No se espera que esto afecte significativamente los features actuales (ángulo de codo, rotación de cadera), pero queda anotado como posible variable a investigar si el modelo mostrara comportamiento inconsistente entre datos de sombra y datos reales.

---

## 15. Reentrenamiento con dataset ampliado y pipeline de dos etapas

### 15.1 Resultados del reentrenamiento

Con el dataset ampliado (538 repeticiones: 288 drive, 250 revés — desde 71 originales), se comparó la arquitectura original (Dense 16→8→1) contra una versión más grande con regularización (Dense 32→Dropout→16→Dropout→8→1), validando ambas con 5-fold cross-validation:

| Golpe | Red pequeña (original) | Red grande (dropout) |
|---|---|---|
| Drive | 89.2% ± 4.7% | 89.6% ± 5.1% |
| Revés | 94.4% ± 2.7% | 94.8% ± 2.7% |

La diferencia entre arquitecturas (+0.4% en ambos casos) es menor que la propia desviación estándar — no es una mejora estadísticamente significativa. **Se mantuvo la arquitectura pequeña original**: la mejora real vino completamente de ampliar el dataset (74-76% → 89-95%), no de aumentar la capacidad del modelo. Esto confirma retroactivamente que la decisión original de usar una red pequeña (Día 23, motivada por el dataset reducido) seguía siendo la correcta incluso con 7 veces más datos.

### 15.2 El problema de out-of-distribution, confirmado de nuevo

Al probar los modelos reentrenados, se confirmó el problema documentado en la sección 10: subir un revés seleccionando "drive" (o viceversa) seguía siendo clasificado como "correcto" con alta confianza. Esto demostró que **el problema no se resuelve con más dataset** — es una limitación estructural: el modelo de drive nunca aprendió a reconocer "¿esto es un drive?", solo "¿estos 8 números parecen un drive correcto?", asumiendo que ya es un drive. Dado que ambos golpes comparten features (ángulo de codo, rotación de cadera), es posible que un revés caiga dentro del rango numérico que el modelo aprendió como "drive correcto".

### 15.3 Solución: identificador de golpe como primera etapa

Con suficientes datos ya disponibles de ambos golpes, se entrenó un tercer modelo — un clasificador binario que predice "¿es drive o revés?", usando los 12 features (incluyendo altura de muñeca, informativa para distinguir el tipo de golpe incluso en drive) sobre las 538 repeticiones combinadas (etiqueta = golpe, ignorando correcto/incorrecto).

**Validación (5-fold cross-validation):** 89.2% ± 5.3% de accuracy.

**Diagnóstico por grupo** (dónde se concentran los errores):

| Grupo | Error |
|---|---|
| Revés incorrecto | 23.7% |
| Drive correcto | 15.2% |
| Drive incorrecto | 8.9% |
| Revés correcto | 3.8% |

El grupo con más error ("revés incorrecto") es consistente con la hipótesis de que un revés mal ejecutado pierde su seña distintiva (elevación de muñeca) y se parece más a un drive. Sin embargo, "drive incorrecto" tuvo *menos* error que "drive correcto", por lo que no hay un patrón limpio de "lo incorrecto es siempre más difícil" — el tamaño desigual de las muestras (los grupos "incorrecto" son 5-6 veces más chicos) probablemente contribuye al ruido más que un patrón conceptual.

**Decisión:** integrar el identificador tal cual (89% de protección es una mejora enorme frente al 0% actual), documentando honestamente que no es perfecto.

### 15.4 Arquitectura final del pipeline (dos etapas)

```
Video → landmarks (MediaPipe) → features (12, con altura de muñeca)
→ Etapa 1: modelo identificador → ¿coincide con el golpe seleccionado?
   → NO: responde error explicando el golpe detectado, termina aquí
   → SI: continúa a Etapa 2
→ Etapa 2: modelo de correcto/incorrecto especifico del golpe → veredicto final
```

Los 3 modelos (`modelo_drive_final.h5`, `modelo_reves_final.h5`, `modelo_identificador_final.h5`) se entrenan automáticamente en cada build de Docker, igual que se decidió en la sección 10 — ningún paso manual adicional al desplegar.

**Validación en producción (pruebas con cámara en vivo):** tras el despliegue, se probaron varios casos reales, incluyendo intentos deliberados de golpe equivocado — el identificador capturó correctamente los mismatches (ej. revés real con "drive" seleccionado, correctamente rechazado) y dejó pasar correctamente los casos que sí coincidían.
