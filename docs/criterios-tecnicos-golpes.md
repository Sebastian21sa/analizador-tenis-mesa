# Criterios técnicos — Análisis de Drive y Revés en Tenis de Mesa

Documento de diseño del proyecto insignia. Basado en criterio de jugador (experiencia propia) y formalizado para su uso con **MediaPipe Pose**.

## Testing y calidad de código (Día 25)

**Refactor:** las funciones `calcular_angulo` y `extraer_features` estaban duplicadas en `entrenar_drive.py` y `entrenar_reves.py`. Se centralizaron en un único módulo compartido (`features.py`), con un parámetro `incluir_altura_muneca` que activa el feature extra del revés — una sola fuente de verdad para ambos modelos.

**Tests unitarios (pytest):** 5 tests cubriendo la lógica más crítica del proyecto:
- `test_angulo_90_grados`, `test_angulo_180_grados`, `test_angulo_0_grados`: validan `calcular_angulo` contra casos geométricos conocidos.
- `test_extraer_features_sin_movimiento`: valida que con una repetición sin movimiento (frames idénticos), min/max/promedio coinciden y el rango es 0.
- `test_extraer_features_incluye_altura_muneca`: valida que el parámetro `incluir_altura_muneca` sí cambia el número de features devueltos (8 vs 12).

**Lección real de ingeniería:** el primer intento de test (`assert angulo_min == angulo_max`) falló por un problema clásico de precisión de punto flotante (`126.86989764584403` vs `126.86989764584402`) — dos cálculos matemáticamente iguales pueden diferir en el último dígito por cómo NumPy ordena las operaciones internamente. Se corrigió usando `pytest.approx()` en vez de igualdad exacta (`==`), la práctica correcta para comparar números decimales en cualquier lenguaje.

**Reproducibilidad:** se detectó que el accuracy variaba fuertemente entre corridas del mismo código (86% → 57%) por la inicialización aleatoria de pesos de la red. Se fijó una semilla (`np.random.seed`, `random.seed`, `tf.random.set_seed`) para que los experimentos sean comparables entre sí.

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

## Próximos pasos

1. Validar este documento grabando 2-3 clips de prueba (ya hecho en el Día 12).
2. Confirmar que los ángulos definidos aquí sí se pueden extraer con claridad del ángulo de cámara elegido.
3. Usar este documento como guía de etiquetado al grabar el dataset el miércoles y jueves.
4. Al grabar, registrar la mano dominante de cada jugador (ej. en el nombre del archivo o en una hoja de registro aparte: `drive_correcto_diestro_01.mp4`, `drive_correcto_zurdo_01.mp4`), para poder aplicar la normalización de lateralidad en el preprocesamiento.

## Arquitectura del proyecto (Día 16)

Dos flujos separados:

**Pipeline de entrenamiento (offline, corre una vez o cuando se agregan más datos):**
Dataset de video → MediaPipe Pose (extrae landmarks) → Preprocesamiento (normalización de lateralidad + cálculo de ángulos) → Entrenamiento (TensorFlow/PyTorch) → Modelo guardado (.h5/.pt)

**Pipeline de inferencia (online, corre en cada uso real de la app):**
Video del usuario → API Flask en Docker → Pose + preprocesamiento (mismo pipeline que en entrenamiento) → Modelo cargado → Respuesta JSON (correcto/incorrecto por golpe) → Frontend muestra el resultado

**Capas transversales (no forman parte de la cadena de una predicción individual):**
- CI/CD con GitHub Actions: corre tests automáticamente en cada `git push`.
- Despliegue: API + Docker en Render/Railway, frontend en Vercel (Semana 6).

## Dataset final (Día 22)

Segmentación automática por velocidad de movimiento de la muñeca (`scipy.signal.find_peaks`), calibrada por separado para cada video:

| Categoría | Umbral (height, distance) | Repeticiones |
|---|---|---|
| Drive correcto | 0.018, 80 | 23 |
| Drive incorrecto | 0.018, 80 | 16 |
| Revés correcto | 0.008, 65 | 20 |
| Revés incorrecto | 0.008, 65 | 12 |
| **Total** | | **71 repeticiones** |

Nota: existe un leve desbalance de clases (revés incorrecto tiene menos ejemplos). A tener en cuenta en el entrenamiento (Semana 4): posible uso de class weighting.

## Resultados de entrenamiento

**Resumen:**
- Drive: 88% de accuracy
- Revés: 86% de accuracy (después de resolver dos problemas reales: features equivocados y class_weight contraproducente)

**Drive (Día 23):** red neuronal densa (16→8→1) con TensorFlow/Keras, entrenada sobre 8 features derivados de ángulos (codo, rotación de cadera) por repetición. Accuracy en test: **88%** (7/8, muestra de prueba pequeña — a validar con más datos/ajuste el Día 27).

### Arquitectura del modelo

Se utilizó una **red neuronal densamente conectada** (Dense / Multi-Layer Perceptron), no una red recurrente (LSTM/RNN) ni convolucional (CNN):

```
Entrada (8 features) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

**Por qué esta arquitectura y no otra:** cada repetición se resume en un vector fijo de 8 números (min/max/rango/promedio de ángulo de codo y rotación de cadera), en vez de usarse la secuencia completa de ~60 frames. Con un dataset pequeño (39 ejemplos para el drive), una red recurrente (pensada para secuencias) tendría alto riesgo de sobreajuste. Una red densa sobre features ya resumidos es la elección apropiada para datos tabulares de tamaño reducido — una decisión consciente de trade-off entre complejidad del modelo y cantidad de datos disponibles, no la opción "por defecto".

- **Capa oculta 1** (16 neuronas, ReLU): combina los 8 features de entrada buscando patrones.
- **Capa oculta 2** (8 neuronas, ReLU): comprime la información de la capa anterior.
- **Capa de salida** (1 neurona, sigmoid): produce un valor entre 0 y 1 interpretado como probabilidad de "golpe correcto".
- **Optimizador:** Adam. **Función de pérdida:** binary crossentropy (estándar para clasificación binaria).

**Revés (Día 24):** misma arquitectura, con un feature adicional específico del revés (altura relativa de la muñeca respecto al hombro, ya que la fase de armado/impacto del revés se distingue por esa altura, no solo por el ángulo del codo). Accuracy final en test: **86%** (6/7).

**Features exactos usados en cada modelo:**

| # | Feature | Drive (8 features) | Revés (12 features) |
|---|---|---|---|
| 1-4 | Ángulo de codo (min, max, rango, promedio) — hombro(12)-codo(14)-muñeca(16) | ✅ | ✅ |
| 5-8 | Rotación de cadera (min, max, rango, promedio) — línea hombros vs línea caderas | ✅ | ✅ |
| 9-12 | Altura relativa de muñeca (min, max, rango, promedio) — muñeca[y] - hombro[y] | ❌ | ✅ |

Los primeros 8 features son idénticos entre ambos golpes. El revés suma 4 features adicionales de altura de muñeca porque, según el criterio documentado, ese golpe se distingue por el desplazamiento vertical de la mano (de la altura del ombligo a la altura del hombro), algo que el ángulo del codo por sí solo no capturaba bien.

**Lección de ingeniería real de este día:** un primer intento con `class_weight` balanceado (para compensar que hay menos ejemplos de "revés incorrecto") causó que el modelo colapsara y predijera siempre la misma clase — con un dataset pequeño y ruidoso, penalizar más fuerte la clase minoritaria puede hacer que el modelo "se rinda" en vez de aprender a distinguir. Quitar el `class_weight` resolvió el problema. Conclusión: `class_weight` no es automáticamente mejor, depende del tamaño y ruido del dataset — se valida empíricamente, no se asume.

**Limitación conocida documentada:** el dataset actual (71 repeticiones totales, propio y auto-etiquetado) es pequeño para estándares de deep learning. Los resultados (88% drive, 86% revés) son alentadores pero no estadísticamente robustos dado el tamaño del set de prueba (7-8 muestras). Mejora futura: ampliar el dataset, en particular la categoría `reves/incorrecto` (la más pequeña, 12 repeticiones).



