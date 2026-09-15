import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [golpe, setGolpe] = useState('drive')
  const [tab, setTab] = useState('archivo')
  const [resultado, setResultado] = useState(null)
  const [cargando, setCargando] = useState(false)
  const [camaraActiva, setCamaraActiva] = useState(false)
  const [grabando, setGrabando] = useState(false)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  async function analizarVideo(archivoOBlob, nombreArchivo) {
    const formData = new FormData()
    formData.append('video', archivoOBlob, nombreArchivo)
    formData.append('golpe', golpe)

    setResultado(null)
    setCargando(true)

    try {
      const respuesta = await fetch(`${import.meta.env.VITE_API_URL}/predecir`,{
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

  function subirYAnalizar(evento) {
    const archivo = evento.target.files[0]
    if (!archivo) return

    const TAMANO_MAXIMO_MB = 20
    const tamanoMB = archivo.size / (1024 * 1024)

    if (tamanoMB > TAMANO_MAXIMO_MB) {
      setResultado({
        error: `El video pesa ${tamanoMB.toFixed(1)} MB. Sube un clip corto de un solo golpe (menos de ${TAMANO_MAXIMO_MB} MB) para mejores resultados.`,
      })
      return
    }

    analizarVideo(archivo, archivo.name)
  }

  async function activarCamara() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      videoRef.current.srcObject = stream
      setCamaraActiva(true)
    } catch (error) {
      alert('No se pudo acceder a la camara. Revisa los permisos del navegador.')
    }
  }

  function iniciarGrabacion() {
    chunksRef.current = []
    const mediaRecorder = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' })

    mediaRecorder.ondataavailable = (evento) => {
      chunksRef.current.push(evento.data)
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' })
      analizarVideo(blob, 'grabacion.webm')
    }

    mediaRecorder.start()
    mediaRecorderRef.current = mediaRecorder
    setGrabando(true)
  }

  function detenerGrabacion() {
    mediaRecorderRef.current.stop()
    setGrabando(false)
  }

  return (
    <div className="pagina">
      <header className="encabezado">
        <div style={{ display: 'flex', alignItems: 'center', maxWidth: 520, margin: '0 auto' }}>
          <div className="marca">
            <img src="/logo.png" alt="PongIQ" style={{ height: 140 }} />
            <p className="eyebrow" style={{ marginTop: 6 }}>Analiza tus golpes y mejora tu técnica de tenis de mesa, Sube un golpe y se analiza en segundos.</p>
          </div>
          <a href="https://web-sebastian.vercel.app" className="volver-portafolio">
            Portafolio →
          </a>
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
            <div className="panel-camara">
              <video ref={videoRef} autoPlay muted playsInline className="video-preview" />
              {!camaraActiva && (
                <button className="boton-primario" onClick={activarCamara}>
                  Activar camara
                </button>
              )}
              {camaraActiva && !grabando && (
                <button className="boton-primario boton-grabar" onClick={iniciarGrabacion}>
                  Grabar golpe
                </button>
              )}
              {grabando && (
                <button className="boton-primario boton-detener" onClick={detenerGrabacion}>
                  Detener y analizar
                </button>
              )}
            </div>
          )}

          {cargando && (
            <div className="cargando">
              <div className="puntos-cargando">
                <span></span><span></span><span></span><span></span>
              </div>
              <p>Analizando tecnica...</p>
              <p className="nota-cargando">Puede tardar hasta un minuto si el servidor estaba inactivo</p>
            </div>
          )}

          {resultado && !cargando && (
            resultado.error ? (
              <div className="marcador aviso">
                <p className="marcador-aviso-titulo">Revisa el golpe seleccionado</p>
                <p className="marcador-aviso-texto">{resultado.error}</p>
              </div>
            ) : (
              <div className={`marcador ${resultado.veredicto}`}>
                <p className="marcador-golpe">{resultado.golpe}</p>
                <p className="marcador-veredicto">{resultado.veredicto}</p>
                <div className="marcador-confianza">
                  <span className="confianza-numero">{(resultado.confianza * 100).toFixed(0)}</span>
                  <span className="confianza-signo">%</span>
                  <span className="confianza-label">confianza</span>
                </div>
              </div>
            )
          )}
        </section>
      </main>
    </div>
  )
}

export default App