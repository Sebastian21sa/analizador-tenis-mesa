import numpy as np
def calcular_angulo(a, b, c):
     """ Calcula el angulo (en grados) formado en el punto b,
       entre los segmentos b-a y b-c. Se usa para medir angulos articulares como hombro-codo-muneca. 
       Args: a, b, c: tuplas o listas (x, y, z) con las coordenadas de cada punto. 
       Solo se usan x e y (2D). Devuelve: float: angulo en grados, entre 0 y 180. """
     a = np.array(a[:2]) 
     b = np.array(b[:2])
     c = np.array(c[:2]) 
     ba = a - b
     bc = c - b
     coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
     coseno = np.clip(coseno, -1.0, 1.0)
     angulo = np.degrees(np.arccos(coseno))
     return angulo

def extraer_features(rep, incluir_altura_muneca=False):
    """ Convierte una repeticion completa (secuencia de frames con landmarks de MediaPipe Pose) 
    en un vector fijo de caracteristicas numericas, listo para un modelo de clasificacion. 
    Indices de MediaPipe Pose usados (jugador diestro): 11 = hombro izquierdo 12 = hombro derecho 14 = codo derecho
    16 = muneca derecha 23 = cadera izquierda 24 = cadera derecha 
    Args: rep: lista de frames, cada frame es una lista de 33 landmarks (x, y, z).
    incluir_altura_muneca (bool): si True, agrega 4 features extra de altura de muneca respecto al hombro (usado en el reves). 
    Devuelve: list[float]: 8 features (drive) o 12 features (reves). """
    angulos_codo = []
    rotaciones_cadera = []
    alturas_muneca = []

    for frame in rep:
        hombro = frame[12]
        codo = frame[14]
        muneca = frame[16]

        angulo_codo = calcular_angulo(hombro, codo, muneca)
        angulos_codo.append(angulo_codo)

        hombro_der = frame[12]
        hombro_izq = frame[11]
        cadera_der = frame[24]
        cadera_izq = frame[23]

        rotacion = (hombro_der[0] - hombro_izq[0]) - (cadera_der[0] - cadera_izq[0])
        rotaciones_cadera.append(rotacion)

        if incluir_altura_muneca:
            altura_relativa = muneca[1] - hombro[1]
            alturas_muneca.append(altura_relativa)

    angulos_codo = np.array(angulos_codo)
    rotaciones_cadera = np.array(rotaciones_cadera)

    features = [
        angulos_codo.min(),
        angulos_codo.max(),
        angulos_codo.max() - angulos_codo.min(),
        angulos_codo.mean(),
        rotaciones_cadera.min(),
        rotaciones_cadera.max(),
        rotaciones_cadera.max() - rotaciones_cadera.min(),
        rotaciones_cadera.mean()
    ]

    if incluir_altura_muneca:
        alturas_muneca = np.array(alturas_muneca)
        features += [
            alturas_muneca.min(),
            alturas_muneca.max(),
            alturas_muneca.max() - alturas_muneca.min(),
            alturas_muneca.mean()
        ]

    return features