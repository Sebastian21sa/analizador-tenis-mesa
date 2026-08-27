import pytest
import numpy as np
from features import calcular_angulo, extraer_features


def test_angulo_90_grados():
    a = (0, 1, 0)
    b = (0, 0, 0)
    c = (1, 0, 0)
    resultado = calcular_angulo(a, b, c)
    assert resultado == pytest.approx(90.0)


def test_angulo_180_grados():
    a = (-1, 0, 0)
    b = (0, 0, 0)
    c = (1, 0, 0)
    resultado = calcular_angulo(a, b, c)
    assert resultado == pytest.approx(180.0)


def test_angulo_0_grados():
    a = (1, 0, 0)
    b = (0, 0, 0)
    c = (2, 0, 0)
    resultado = calcular_angulo(a, b, c)
    assert resultado == pytest.approx(0.0, abs=1e-3)


def crear_frame_falso(hombro_der=(0.5, 0.3, 0), hombro_izq=(0.3, 0.3, 0),
                       codo_der=(0.6, 0.5, 0), muneca_der=(0.5, 0.7, 0),
                       cadera_der=(0.5, 0.6, 0), cadera_izq=(0.35, 0.6, 0)):
    frame = [(0, 0, 0)] * 33
    frame[11] = hombro_izq
    frame[12] = hombro_der
    frame[14] = codo_der
    frame[16] = muneca_der
    frame[23] = cadera_izq
    frame[24] = cadera_der
    return frame


def test_extraer_features_sin_movimiento():
    frame_fijo = crear_frame_falso()
    repeticion = [frame_fijo] * 10

    features = extraer_features(repeticion, incluir_altura_muneca=False)

    angulo_min = features[0]
    angulo_max = features[1]
    angulo_rango = features[2]
    angulo_promedio = features[3]

    assert angulo_min == pytest.approx(angulo_max)
    assert angulo_max == pytest.approx(angulo_promedio)
    assert angulo_rango == pytest.approx(0.0, abs=1e-9)
    assert len(features) == 8


def test_extraer_features_incluye_altura_muneca():
    frame_fijo = crear_frame_falso()
    repeticion = [frame_fijo] * 10

    features = extraer_features(repeticion, incluir_altura_muneca=True)

    assert len(features) == 12