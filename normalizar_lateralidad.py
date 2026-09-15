def espejar_landmarks(frame):
    """
    Espeja horizontalmente un frame de landmarks, intercambiando izquierda/derecha,
    para normalizar videos de jugadores zurdos a la misma convencion que los diestros
    (mano dominante = lado derecho).
    """
    frame_espejado = list(frame)
    pares_espejo = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]

    for izq, der in pares_espejo:
        frame_espejado[izq], frame_espejado[der] = frame[der], frame[izq]

    frame_espejado = [(1 - x, y, z) for (x, y, z) in frame_espejado]

    return frame_espejado


def espejar_repeticion(repeticion):
    """Aplica el espejo a todos los frames de una repeticion completa."""
    return [espejar_landmarks(frame) for frame in repeticion]
