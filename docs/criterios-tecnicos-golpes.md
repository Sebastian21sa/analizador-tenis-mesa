# Criterios técnicos — Análisis de Drive y Revés en Tenis de Mesa

Documento de diseño del proyecto tecnica de golpes basicos de tenis de mesa. Basados en mi criterio de jugador (experiencia propia) y formalizado para su uso con **MediaPipe Pose**.

## Landmarks de MediaPipe Pose que vamos a usar

MediaPipe Pose devuelve 33 puntos del cuerpo. Para este proyecto, los relevantes son:

| Punto del cuerpo | Índice — lado derecho | Índice — lado izquierdo |
|---|---|---|
| Hombro | 12 | 11 |
| Codo | 14 | 13 |
| Muñeca | 16 | 15 |
| Cadera | 24 | 23 |
| Rodilla | 26 | 25 |
| Tobillo | 28 | 27 |
| Nariz (referencia de cabeza) | 0 | 0 |

**Cómo se usan estos dos lados junto con la normalización:**
- Para un jugador **diestro**: se toman directamente los índices de la columna "lado derecho" (12, 14, 16, 24, 26, 28) como brazo/pierna dominante.
- Para un jugador **zurdo**: se toman los índices de la columna "lado izquierdo" (11, 13, 15, 23, 25, 27) como dominante, y luego se invierten (espejo en el eje X) para que las coordenadas queden en la misma convención que un diestro.
- Después de este paso, el resto del pipeline (cálculo de ángulos, entrenamiento del modelo) trabaja siempre sobre la convención de "lado derecho = dominante", sin importar de qué lado jugaba originalmente la persona en el video.

## Ángulos y medidas derivadas (features calculables)

- **Ángulo de codo**: ángulo formado por hombro–codo–muñeca (del brazo dominante).
- **Ángulo de rotación de tronco/cadera**: ángulo entre la línea que une ambas caderas y la línea que une ambos hombros, comparado contra la posición base (neutral).
- **Transferencia de peso**: desplazamiento lateral (eje X) del punto medio entre ambos tobillos, o de la cadera dominante, entre fases.
- **Estabilidad del codo**: variación de posición (X, Y) del codo entre la fase de preparación y la fase de impacto — debería mantenerse casi fija según tu criterio.

**Limitación conocida:** la rotación de la muñeca/antebrazo (palma hacia arriba/abajo) no es medible de forma confiable solo con MediaPipe Pose, porque el modelo da posición del punto, no orientación de la palma. Queda documentado como mejora futura (posible incorporación de MediaPipe Hands).

## Manejo de jugadores zurdos (normalización de lateralidad)

El dataset va a incluir tanto jugadores diestros como zurdos. En vez de mantener dos conjuntos de reglas paralelos (uno por cada lado), se normaliza todo a una sola convención:

- Cada clip se etiqueta con un metadato `mano_dominante: "derecha" | "izquierda"`.
- En el preprocesamiento, los clips de jugadores **zurdos se invierten horizontalmente** (espejo del eje X de cada landmark) antes de calcular ángulos y features.
- Resultado: el modelo entrena sobre un solo patrón unificado (equivalente a "todos diestros"), sin necesitar aprender dos versiones espejadas del mismo movimiento. Esto también duplica efectivamente el aprovechamiento del dataset entre ambos lados.
- Los índices de landmarks descritos en este documento (12, 14, 16 para el lado derecho) se usan como la convención canónica **después** de esta normalización.

---

## Posición base (stance neutral)

Punto de partida de ambos golpes.

- Piernas abiertas al ancho de los hombros o más (distancia entre tobillos ≈ distancia entre hombros).
- Jugador derecho: pie izquierdo levemente adelantado respecto al derecho (y viceversa para zurdos).
- Ángulo de codo ≈ 90° (hombro–codo–muñeca).
- Cabeza orientada al frente (nariz alineada con el eje central de hombros).

---

## DRIVE

| Fase | Descripción | Landmarks / ángulos clave |
|---|---|---|
| **Fase 0 — Posición base** | Ángulo de codo ≈ 90°, stance neutral | Ángulo codo, posición cadera neutral |
| **Fase 1 — Armado (backswing)** | Rotación de tronco/cadera hacia el lado dominante (derecha si es diestro); transferencia de peso a la pierna del lado dominante | Rotación cadera-hombro, desplazamiento lateral de tobillo/cadera dominante |
| **Fase 2 — Impacto** | Retorno de cadera y peso hacia el lado no dominante mientras el antebrazo avanza al frente; **el codo NO se debe desplazar de su posición** | Ángulo codo estable (comparado con Fase 0), rotación cadera-hombro invertida, desplazamiento de peso |
| **Retorno** | Vuelve a Fase 0 | — |

**Errores comunes (label = incorrecto):**
- Desplazamiento del codo fuera de su posición estable
- Movimiento aislado de la muñeca/raqueta sin acompañamiento del cuerpo
- Ausencia de transferencia de peso (cadera no rota)
- Golpe ejecutado solo con el brazo (sin rotación de tronco)
- Rango de movimiento exagerado

---

## REVÉS

| Fase | Descripción | Landmarks / ángulos clave |
|---|---|---|
| **Fase 0 — Posición base** | Igual que en el drive | Ángulo codo, stance neutral |
| **Fase 1 — Armado** | Antebrazo/muñeca se dirige hacia el cuerpo, aprox. altura del ombligo | Posición de muñeca relativa al torso (altura, eje Y) |
| **Fase 2 — Impacto** | Antebrazo se desplaza hacia la altura del hombro dominante; peso del cuerpo se proyecta hacia adelante | Posición de muñeca relativa al hombro, desplazamiento de peso hacia adelante (eje Y de cadera/tobillo) |
| **Retorno** | Vuelve a Fase 0 | — |

**Errores comunes (label = incorrecto):**
- Mismos errores generales que el drive (codo, brazo aislado, falta de peso, rango exagerado)

---


## Arquitectura del proyecto 

Dos flujos separados:

**Pipeline de entrenamiento (offline, corre una vez o cuando se agregan más datos):**
Dataset de video → MediaPipe Pose (extrae landmarks) → Preprocesamiento (normalización de lateralidad + cálculo de ángulos) → Entrenamiento (TensorFlow/PyTorch) → Modelo guardado (.h5/.pt)

**Pipeline de inferencia (online, corre en cada uso real de la app):**
Video del usuario → API Flask en Docker → Pose + preprocesamiento (mismo pipeline que en entrenamiento) → Modelo cargado → Respuesta JSON (correcto/incorrecto por golpe) → Frontend muestra el resultado

