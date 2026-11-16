import { useState } from 'react'
import { supabase } from './supabaseClient'

function traducirError(mensaje) {
  if (mensaje.includes('Invalid login credentials')) {
    return 'Correo o contraseña incorrectos.'
  }
  if (mensaje.includes('Email not confirmed')) {
    return 'Debes confirmar tu correo antes de entrar. Revisa tu bandeja de entrada.'
  }
  if (mensaje.includes('already registered')) {
    return 'Ya existe una cuenta con ese correo.'
  }
  if (mensaje.includes('at least')) {
    return 'La contraseña debe tener al menos 6 caracteres.'
  }
  return 'No se pudo completar la acción. Intenta de nuevo.'
}

function Login({ onCerrar }) {
  const [modo, setModo] = useState('entrar')
  const [correo, setCorreo] = useState('')
  const [clave, setClave] = useState('')
  const [enviando, setEnviando] = useState(false)
  const [error, setError] = useState(null)
  const [aviso, setAviso] = useState(null)

  async function enviar(evento) {
    evento.preventDefault()
    setError(null)
    setAviso(null)
    setEnviando(true)

    if (modo === 'registro') {
      const { data, error: fallo } = await supabase.auth.signUp({
        email: correo,
        password: clave,
      })

      if (fallo) {
        setError(traducirError(fallo.message))
      } else if (!data.session) {
        setAviso('Te enviamos un correo para confirmar tu cuenta. Revisa tu bandeja de entrada.')
      } else {
        onCerrar()
      }
    } else {
      const { error: fallo } = await supabase.auth.signInWithPassword({
        email: correo,
        password: clave,
      })

      if (fallo) {
        setError(traducirError(fallo.message))
      } else {
        onCerrar()
      }
    }

    setEnviando(false)
  }

  return (
    <div className="login-fondo" onClick={onCerrar}>
      <div className="login-caja" onClick={(evento) => evento.stopPropagation()}>
        <button className="login-cerrar" onClick={onCerrar} aria-label="Cerrar">
          ×
        </button>

        <h2 className="login-titulo">
          {modo === 'entrar' ? 'Iniciar sesión' : 'Crear cuenta'}
        </h2>

        <form onSubmit={enviar} className="login-formulario">
          <label className="login-campo">
            <span>Correo</span>
            <input
              type="email"
              value={correo}
              onChange={(evento) => setCorreo(evento.target.value)}
              required
              autoComplete="email"
            />
          </label>

          <label className="login-campo">
            <span>Contraseña</span>
            <input
              type="password"
              value={clave}
              onChange={(evento) => setClave(evento.target.value)}
              required
              minLength={6}
              autoComplete={modo === 'entrar' ? 'current-password' : 'new-password'}
            />
          </label>

          {error && <p className="login-error">{error}</p>}
          {aviso && <p className="login-aviso">{aviso}</p>}

          <button type="submit" className="boton-primario" disabled={enviando}>
            {enviando
              ? 'Un momento...'
              : modo === 'entrar'
                ? 'Entrar'
                : 'Crear cuenta'}
          </button>
        </form>

        <p className="login-cambio">
          {modo === 'entrar' ? '¿No tienes cuenta?' : '¿Ya tienes cuenta?'}{' '}
          <button
            type="button"
            className="login-enlace"
            onClick={() => {
              setModo(modo === 'entrar' ? 'registro' : 'entrar')
              setError(null)
              setAviso(null)
            }}
          >
            {modo === 'entrar' ? 'Crear una' : 'Entrar'}
          </button>
        </p>
      </div>
    </div>
  )
}

export default Login