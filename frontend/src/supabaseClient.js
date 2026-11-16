import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const llave = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY

if (!url || !llave) {
  console.warn(
    'Supabase no está configurado: faltan VITE_SUPABASE_URL o VITE_SUPABASE_PUBLISHABLE_KEY.'
  )
}

export const supabase = url && llave ? createClient(url, llave) : null