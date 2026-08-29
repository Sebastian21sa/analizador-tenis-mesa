import { useState } from 'react'
import './App.css'

function App() {
  const [golpe, setGolpe] = useState('drive')
  const [tab, setTab] = useState('archivo')
  const [resultado, setResultado] = useState(null)
  const [cargando, setCargando] = useState(false)

  async function subirYAnalizar(evento) {
    const archivo = evento.target.files[0]
    if (!archivo) return

    const formData = new FormData()
    formData.append('video', archivo)
    formData.append('golpe', golpe)

    setResultado(null)
    setCargando(true)

    try {
      const respuesta = await fetch('http://127.0.0.1:5000/predecir', {
        method: 'POST',
        body: formData,
      })
      const datos = await respuesta.json()
      setResultado(datos)
    } catch (error) {
      setResultado({ error: 'No se pudo conectar con el servidor' })
    } finally {
      setCargando(false)
    }
  }

  return (
    <div className="contenedor">
      <h1>Analizador de Tecnica</h1>
      <p>Tenis de mesa - Drive y Reves</p>

      <div className="selector-golpe">
        <label>
          <input
            type="radio"
            name="golpe"
            value="drive"
            checked={golpe === 'drive'}
            onChange={(e) => setGolpe(e.target.value)}
          />
          Drive
        </label>
        <label>
          <input
            type="radio"
            name="golpe"
            value="reves"
            checked={golpe === 'reves'}
            onChange={(e) => setGolpe(e.target.value)}
          />
          Reves
        </label>
      </div>

      <div className="tabs">
        <button
          className={tab === 'archivo' ? 'tab-btn activo' : 'tab-btn'}
          onClick={() => setTab('archivo')}
        >
          Subir video
        </button>
        <button
          className={tab === 'camara' ? 'tab-btn activo' : 'tab-btn'}
          onClick={() => setTab('camara')}
        >
          Grabar con camara
        </button>
      </div>

      {tab === 'archivo' && (
        <div className="panel">
          <input type="file" accept="video/*" onChange={subirYAnalizar} />
        </div>
      )}

      {resultado && (
        <div className={resultado.error ? 'resultado incorrecto' : `resultado ${resultado.veredicto}`}>
          {resultado.error
            ? `Error: ${resultado.error}`
            : `Golpe: ${resultado.golpe} | Veredicto: ${resultado.veredicto} (confianza: ${(resultado.confianza * 100).toFixed(0)}%)`}
        </div>
      )}

      {cargando && <div className="resultado">Analizando...</div>}
    </div>
  )
}

export default App