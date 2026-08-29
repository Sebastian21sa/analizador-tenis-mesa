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
    <div className="pagina">
      <header className="encabezado">
        <div className="marca">
          <svg className="icono-marca" viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="8" r="3" fill="currentColor" />
            <circle cx="8" cy="20" r="3" fill="currentColor" />
            <circle cx="32" cy="20" r="3" fill="currentColor" />
            <circle cx="20" cy="32" r="3" fill="currentColor" />
            <line x1="20" y1="8" x2="8" y2="20" stroke="currentColor" strokeWidth="1.5" />
            <line x1="20" y1="8" x2="32" y2="20" stroke="currentColor" strokeWidth="1.5" />
            <line x1="8" y1="20" x2="20" y2="32" stroke="currentColor" strokeWidth="1.5" />
            <line x1="32" y1="20" x2="20" y2="32" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          <div>
            <h1>Tecnica</h1>
            <p className="eyebrow">Analisis de drive y reves</p>
          </div>
        </div>
      </header>

      <main className="contenido">
        <section className="tarjeta">
          <div className="selector-golpe">
            <button
              className={golpe === 'drive' ? 'golpe-btn activo' : 'golpe-btn'}
              onClick={() => setGolpe('drive')}
            >
              Drive
            </button>
            <button
              className={golpe === 'reves' ? 'golpe-btn activo' : 'golpe-btn'}
              onClick={() => setGolpe('reves')}
            >
              Reves
            </button>
          </div>

          <div className="tabs">
            <button
              className={tab === 'archivo' ? 'tab activo' : 'tab'}
              onClick={() => setTab('archivo')}
            >
              Subir video
            </button>
            <button
              className={tab === 'camara' ? 'tab activo' : 'tab'}
              onClick={() => setTab('camara')}
            >
              Grabar con camara
            </button>
          </div>

          {tab === 'archivo' && (
            <label className="zona-subida">
              <input type="file" accept="video/*" onChange={subirYAnalizar} hidden />
              <svg className="icono-subida" viewBox="0 0 24 24" fill="none">
                <path d="M12 16V4M12 4L7 9M12 4L17 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 16V18C4 19.1046 4.89543 20 6 20H18C19.1046 20 20 19.1046 20 18V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span>Arrastra tu video o haz clic para seleccionar</span>
            </label>
          )}

          {tab === 'camara' && (
            <div className="zona-subida proximamente">
              <span>Grabacion con camara disponible manana (Dia 32)</span>
            </div>
          )}

          {cargando && (
            <div className="cargando">
              <div className="puntos-cargando">
                <span></span><span></span><span></span><span></span>
              </div>
              <p>Analizando tecnica...</p>
            </div>
          )}

          {resultado && !cargando && (
            <div className={resultado.error ? 'marcador error' : `marcador ${resultado.veredicto}`}>
              {resultado.error ? (
                <p className="marcador-error-texto">{resultado.error}</p>
              ) : (
                <>
                  <p className="marcador-golpe">{resultado.golpe}</p>
                  <p className="marcador-veredicto">{resultado.veredicto}</p>
                  <div className="marcador-confianza">
                    <span className="confianza-numero">{(resultado.confianza * 100).toFixed(0)}</span>
                    <span className="confianza-signo">%</span>
                    <span className="confianza-label">confianza</span>
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App