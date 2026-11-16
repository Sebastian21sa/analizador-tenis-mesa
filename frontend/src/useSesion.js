import { useEffect, useState } from 'react'
import { supabase } from './supabaseClient'

export function useSesion() {
  const [sesion, setSesion] = useState(null)
  const [cargandoSesion, setCargandoSesion] = useState(Boolean(supabase))

  useEffect(() => {
    if (!supabase) return

    supabase.auth.getSession().then(({ data }) => {
      setSesion(data.session)
      setCargandoSesion(false)
    })

    const { data: suscripcion } = supabase.auth.onAuthStateChange(
      (_evento, nuevaSesion) => {
        setSesion(nuevaSesion)
      }
    )

    return () => suscripcion.subscription.unsubscribe()
  }, [])

  return { sesion, cargandoSesion }
}