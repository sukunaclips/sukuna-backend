"""SUKUNA CLIPS backend: FastAPI + ffmpeg + faster-whisper."""
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

OUT = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUT.mkdir(exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny / base / small
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI(title="SUKUNA CLIPS API")
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/downloads", StaticFiles(directory=OUT), name="downloads")

HOOKS = [
    "secreto", "nunca", "increíble", "error", "importante", "verdad", "dinero",
    "cómo", "por qué", "gratis", "mejor", "peor", "problema", "truco", "clave",
    "secret", "never", "mistake", "truth", "money", "how", "why", "best", "worst",
]

STYLES = {  # estilos ASS (colores en BGR)
    "classic": "FontSize=18,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Bold=1",
    "bold": "FontSize=28,PrimaryColour=&H6BE5F9&,OutlineColour=&H000000&,Outline=3,Bold=1",
    "neon": "FontSize=24,PrimaryColour=&HFF7ABD&,OutlineColour=&H3A0A5A&,Outline=3,Bold=1",
    "minimal": "FontSize=16,PrimaryColour=&HFFFFFF&,Outline=1,Bold=0",
}


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:] or "ffmpeg falló")
    return r.stdout


def video_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)])
    return float(out.strip())


_model = None


def transcribe(path):
    """Devuelve [(inicio, fin, texto)]. Usa poca memoria: audio mono 16 kHz."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    wav = path.with_suffix(".wav")
    try:
        run(["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-t", "1800", str(wav)])
        if _model is None:
            _model = WhisperModel(WHISPER_MODEL, compute_type="int8", cpu_threads=1)
        segments, _ = _model.transcribe(str(wav), vad_filter=True, beam_size=1)
        return [(s.start, s.end, s.text.strip()) for s in segments]
    except Exception:
        return []
    finally:
        wav.unlink(missing_ok=True)


def score_window(segs, a, b):
    text = " ".join(t for s, e, t in segs if e > a and s < b)
    words = len(text.split())
    if not words:
        return 0, ""
    low = text.lower()
    pts = words / (b - a) * 10                      # densidad de habla
    pts += 4 * (text.count("?") + text.count("!"))  # preguntas / énfasis
    pts += 3 * sum(h in low for h in HOOKS)         # palabras gancho
    return pts, text


def pick_clips(segs, total, dur, n):
    dur = min(dur, total)
    step = max(dur / 3, 5)
    cands, t = [], 0.0
    while t + dur <= total + 0.01:
        if segs:
            sc, txt = score_window(segs, t, t + dur)
        else:
            sc, txt = 1, ""
        cands.append((sc, t, txt))
        t += step
    if not segs:  # sin transcripción: reparte clips a lo largo del video
        gap = max((total - dur) / max(n - 1, 1), dur)
        cands = [(1, min(i * gap, max(total - dur, 0)), "") for i in range(n)]
    cands.sort(reverse=True)
    chosen = []
    for sc, st, txt in cands:
        if all(abs(st - c[1]) >= dur for c in chosen):
            chosen.append((sc, st, txt))
        if len(chosen) == n:
            break
    top = max((c[0] for c in chosen), default=1) or 1
    chosen.sort(key=lambda c: c[1])
    return [
        {"start": round(st, 1), "end": round(min(st + dur, total), 1),
         "score": int(55 + 43 * (sc / top)), "text": txt}
        for sc, st, txt in chosen
    ]


def fmt(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def write_srt(segs, start, end, path):
    lines, i = [], 1
    for s, e, txt in segs:
        if e <= start or s >= end:
            continue
        a, b = max(s, start) - start, min(e, end) - start
        lines.append(f"{i}\n{fmt(a)} --> {fmt(b)}\n{txt}\n")
        i += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    return bool(lines)


def ratio_filter(ratio):
    if ratio == "9:16":
        return "crop='min(iw,ih*9/16)':ih,scale=720:1280"
    if ratio == "1:1":
        return "crop='min(iw,ih)':'min(iw,ih)',scale=720:720"
    return ("scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2")


def cut_clip(src, clip, out, ratio, srt, style):
    vf = ratio_filter(ratio)
    cwd = None
    if srt and srt.exists() and srt.stat().st_size:
        vf += f",subtitles={srt.name}:force_style='{STYLES.get(style, STYLES['classic'])}'"
        cwd = srt.parent
    run(["ffmpeg", "-y", "-ss", str(clip["start"]), "-i", str(src.resolve()),
         "-t", str(clip["end"] - clip["start"]), "-vf", vf,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-threads", "1",
         "-c:a", "aac", "-movflags", "+faststart", str(out.resolve())], cwd=cwd)


@app.get("/api/health")
def health():
    return {"ok": True, "ffmpeg": shutil.which("ffmpeg") is not None}


@app.post("/api/process")
def process(
    request: Request,
    file: UploadFile = File(...),
    ratio: str = Form("9:16"),
    clips: int = Form(5),
    duration: int = Form(30),
    captions: bool = Form(True),
    style: str = Form("classic"),
):
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg no está instalado en el servidor.")
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Formato no soportado. Usa MP4, MOV, MKV o WEBM.")

    job = uuid.uuid4().hex[:12]
    work = OUT / job
    work.mkdir()
    src = work / f"source{ext}"
    try:
        with open(src, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if src.stat().st_size > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"El archivo supera {MAX_MB} MB.")

        total = video_duration(src)
        segs = transcribe(src) if captions else []
        picks = pick_clips(segs, total, duration, clips)

        zpath = work / "clips.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for i, c in enumerate(picks, 1):
                out = work / f"clip_{i}.mp4"
                srt = work / f"clip_{i}.srt"
                has_subs = captions and segs and write_srt(segs, c["start"], c["end"], srt)
                cut_clip(src, c, out, ratio, srt if has_subs else None, style)
                z.write(out, out.name)
                c["subtitle"] = "Incluidos" if has_subs else "Sin subtítulos"
                c["file"] = f"{request.base_url}downloads/{job}/{out.name}"
                c["text"] = c["text"][:220] or "Clip generado."
        src.unlink(missing_ok=True)
        return {"clips": picks, "download": f"{request.base_url}downloads/{job}/clips.zip"}
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Error procesando el video: {e}")


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


PAGE = r'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#090812">
<title>SUKUNA CLIPS V2</title>
<style>
:root{color-scheme:dark;--bg:#090812;--panel:#151321;--purple:#9565ff;--text:#fff;--muted:#a39caf;--border:#302943;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:Inter,Arial,sans-serif;min-height:100vh}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);background:#100e19;z-index:11}
button,input,select{font:inherit}button{cursor:pointer}
header{height:70px;background:#100e19;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 22px;position:sticky;top:env(safe-area-inset-top,0px);z-index:10}
.logo{font-size:21px;font-weight:900;letter-spacing:1px}.logo span{color:var(--purple)}
.version{font-size:10px;background:#302044;color:#c9aaff;padding:5px 8px;border-radius:6px;margin-left:5px}
.header-right{display:flex;gap:10px;align-items:center}
.avatar{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#a878ff,#5122a6);display:grid;place-items:center;font-weight:bold}
.layout{display:flex;min-height:calc(100vh - 70px)}
aside{width:220px;padding:25px 13px;border-right:1px solid var(--border);background:#100e19;flex-shrink:0}
.side-label{color:#716b7e;font-size:10px;font-weight:bold;letter-spacing:1.5px;margin:22px 12px 10px}
.nav{display:flex;gap:12px;align-items:center;padding:13px 12px;margin:5px 0;border-radius:9px;color:#aaa3b8;font-size:14px;cursor:pointer}
.nav:hover,.nav.active{color:#fff;background:#302343}
.nav.active{border-left:3px solid var(--purple)}
main{flex:1;min-width:0;padding:30px;max-width:1400px}
.page{display:none}.page.active{display:block}
.heading{display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:25px;flex-wrap:wrap}
h1{font-size:27px}h2{font-size:18px;margin-bottom:17px}h3{font-size:15px}
.muted{color:var(--muted);font-size:13px;line-height:1.6}
.purple{color:#b69aff}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:21px;margin-bottom:20px}
.hero{background:linear-gradient(120deg,#25143c,#171226 65%,#11101b);border:1px solid #503478;border-radius:16px;padding:28px;margin-bottom:23px}
.hero h1{font-size:30px;margin-bottom:12px}.hero p{color:#c0b6d0;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin-bottom:20px}
.stat{background:var(--panel);border:1px solid var(--border);padding:18px;border-radius:12px}
.stat strong{font-size:25px;display:block;margin:8px 0}.stat small{color:var(--muted)}
.input{width:100%;background:#0d0b15;border:1px solid #3a3150;border-radius:9px;padding:14px;color:#fff;margin:9px 0 14px;outline:none}
.input:focus{border-color:var(--purple)}
label{font-size:13px;color:#c6bfd0}
.btn{background:linear-gradient(110deg,#a176ff,#7042d7);color:#fff;border:0;padding:13px 18px;border-radius:9px;font-weight:bold}
.btn:hover{filter:brightness(1.13)}.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.secondary{background:#302a40;border:1px solid #504365}
.btn.full{width:100%}.btn.small{padding:9px 12px;font-size:12px}
.tabs{display:flex;gap:8px;margin:18px 0}
.tab{border:1px solid var(--border);background:#211c2e;color:#aaa;padding:11px 14px;border-radius:8px}
.tab.active{background:#53338c;color:#fff}
.hidden{display:none!important}
.upload{border:2px dashed #574078;padding:30px 15px;text-align:center;border-radius:12px;margin:15px 0}
.upload input{max-width:100%;margin-top:17px}
.preview{width:100%;max-height:400px;border-radius:12px;margin-top:15px;background:#000}
#youtubeFrame{width:100%;height:300px;border:0;border-radius:12px;margin-top:15px}
.notice{padding:13px;border-radius:9px;background:#29203b;color:#cdb8ff;font-size:12px;line-height:1.6;margin:15px 0}
.status{padding:12px;color:#c4a7ff;font-size:13px;line-height:1.6}
select{background:#0d0b15;color:#fff;border:1px solid #403651;padding:12px;border-radius:8px;width:100%;margin:9px 0 16px}
.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.clip{background:#211c30;border:1px solid #45345f;padding:16px;border-radius:12px;min-width:0}
.clip p{color:#aaa2b8;font-size:12px;line-height:1.6;margin:10px 0;overflow-wrap:anywhere}
.score{color:#c4a1ff;font-weight:bold;margin:8px 0}
.tag{display:inline-block;padding:5px 8px;background:#35234d;color:#cbb3ff;border-radius:6px;font-size:10px;margin:3px}
.progress{height:7px;background:#332c42;border-radius:10px;overflow:hidden;margin:12px 0}
.progress div{height:100%;width:0%;background:linear-gradient(90deg,#7848dd,#c3a4ff);transition:width .3s}
.check{display:flex;align-items:center;gap:9px;margin:12px 0;font-size:13px;color:#c5bdcf}
.feature{padding:16px;background:#201b2c;border:1px solid var(--border);border-radius:11px}
.feature b{display:block;margin-bottom:7px}
.bottom-note{text-align:center;color:#6f687b;padding:25px;font-size:11px}
.empty{padding:30px;text-align:center;color:#898194;font-size:13px}
.mnav{display:none}
@media(max-width:760px){aside{display:none}.mnav{display:flex;overflow-x:auto;gap:6px;padding:8px 13px;background:#100e19;border-bottom:1px solid var(--border);position:sticky;top:calc(70px + env(safe-area-inset-top,0px));z-index:9}.mnav .nav{white-space:nowrap;padding:9px 12px;margin:0;font-size:13px;border-left:0!important;background:#211c2e}.mnav .nav.active{background:#53338c;color:#fff}main{padding:18px 13px}header{padding:0 14px}.hero h1{font-size:24px}.grid{grid-template-columns:1fr}.stat{padding:12px 8px}.stat strong{font-size:19px}.stat small{font-size:10px}#youtubeFrame{height:220px}}
</style>
</head>
<body>
<header>
 <div class="logo">SUKUNA <span>CLIPS</span><span class="version">V2</span></div>
 <div class="header-right"><span class="muted">AI STUDIO</span><div class="avatar">S</div></div>
</header>
<div class="mnav">
 <div class="nav active" data-page="home">✦ Inicio</div>
 <div class="nav" data-page="create">✂ Crear clips</div>
 <div class="nav" data-page="projects">▣ Proyectos</div>
 <div class="nav" data-page="captions">▤ Subtítulos</div>
 <div class="nav" data-page="settings">⚙ Configuración</div>
 <div class="nav" data-page="plans">♛ Planes</div>
 <div class="nav" data-page="help">? Ayuda</div>
</div>
<div class="layout">
<aside>
 <div class="side-label">WORKSPACE</div>
 <div class="nav active" data-page="home">✦ Inicio</div>
 <div class="nav" data-page="create">✂ Crear clips</div>
 <div class="nav" data-page="projects">▣ Mis proyectos</div>
 <div class="side-label">HERRAMIENTAS</div>
 <div class="nav" data-page="captions">▤ Subtítulos</div>
 <div class="nav" data-page="settings">⚙ Configuración</div>
 <div class="side-label">SUKUNA</div>
 <div class="nav" data-page="plans">♛ Planes PRO</div>
 <div class="nav" data-page="help">? Ayuda</div>
</aside>
<main>

<section class="page active" id="home">
 <div class="hero">
  <h1>Bienvenido a <span class="purple">SUKUNA</span></h1>
  <p>Transforma tus videos largos en clips cortos, atractivos y preparados para tus redes sociales.</p><br>
  <button class="btn" onclick="navigate('create')">+ Crear nuevo proyecto</button>
 </div>
 <div class="stats">
  <div class="stat"><small>Proyectos guardados</small><strong id="projectCount">0</strong><small>En este dispositivo</small></div>
  <div class="stat"><small>Clips generados</small><strong id="clipCount">0</strong><small>Historial local</small></div>
  <div class="stat"><small>Versión</small><strong>2.0</strong><small>SUKUNA STUDIO</small></div>
 </div>
 <div class="card">
  <h2>🚀 ¿Qué quieres crear?</h2>
  <div class="grid">
   <div class="feature"><b>🎬 Clips automáticos</b><p class="muted">Encuentra momentos interesantes mediante análisis de transcripciones.</p></div>
   <div class="feature"><b>📝 Subtítulos</b><p class="muted">Transcribe tu video y prepara subtítulos.</p></div>
   <div class="feature"><b>📱 Formato vertical</b><p class="muted">Prepara contenido en formato 9:16.</p></div>
   <div class="feature"><b>🔥 Viral Score</b><p class="muted">Obtén una puntuación orientativa para tus clips.</p></div>
  </div>
 </div>
</section>

<section class="page" id="create">
 <div class="heading">
  <div><h1>Crear clips</h1><p class="muted">Tu estudio de edición con inteligencia artificial.</p></div>
  <span class="tag">SUKUNA AI</span>
 </div>
 <div class="card">
  <h2>1. Importar video</h2>
  <div class="tabs">
   <button class="tab active" id="ytTab" onclick="switchImport('youtube')">YouTube URL</button>
   <button class="tab" id="fileTab" onclick="switchImport('file')">Subir archivo</button>
  </div>
  <div id="youtubeBox">
   <label>Enlace o código de YouTube</label>
   <input class="input" id="youtubeInput" placeholder="Pega el enlace o código del video">
   <button class="btn full" onclick="previewYoutube()">Identificar video</button>
   <div id="youtubeResult"></div>
   <div class="notice">El enlace sirve para identificar y visualizar el video. Para crear clips debes subir el archivo original o utilizar una fuente de video autorizada.</div>
  </div>
  <div id="fileBox" class="hidden">
   <div class="upload">
    <h2>📁 Selecciona tu video</h2>
    <p class="muted">MP4, MOV, MKV o WEBM</p>
    <input type="file" id="videoInput" accept="video/mp4,video/quicktime,video/x-matroska,video/webm">
   </div>
   <video id="localPreview" class="preview hidden" controls></video>
   <p class="muted" id="fileInfo"></p>
   <label class="check"><input type="checkbox" id="rights">Tengo autorización para procesar este video.</label>
  </div>
 </div>

 <div class="card">
  <h2>2. Configurar clips</h2>
  <div class="grid">
   <div><label>Formato del video</label>
    <select id="ratio"><option value="9:16">Vertical 9:16</option><option value="16:9">Horizontal 16:9</option><option value="1:1">Cuadrado 1:1</option></select></div>
   <div><label>Cantidad de clips</label>
    <select id="clipAmount"><option value="3">3 clips</option><option value="5" selected>5 clips</option><option value="10">10 clips</option></select></div>
   <div><label>Duración de clips</label>
    <select id="duration"><option value="15">15 segundos</option><option value="30" selected>30 segundos</option><option value="45">45 segundos</option><option value="60">60 segundos</option></select></div>
   <div><label>Estilo de subtítulos</label>
    <select id="captionStyle"><option value="classic">Clásico</option><option value="bold">Grande y llamativo</option><option value="neon">Neón</option><option value="minimal">Minimalista</option></select></div>
  </div>
  <label class="check"><input type="checkbox" id="captions" checked>Generar subtítulos automáticos</label>
  <label class="check"><input type="checkbox" id="viral" checked>Analizar potencial viral</label>
  <label class="check"><input type="checkbox" id="faceTracking" checked>Solicitar seguimiento del hablante</label>
  <button class="btn full" id="generate" onclick="generateClips()">✨ GENERAR CLIPS CON SUKUNA AI</button>
  <div class="progress"><div id="progressBar"></div></div>
  <div class="status" id="status">Esperando tu video...</div>
 </div>

 <div class="card">
  <h2>3. Resultados</h2>
  <div id="results" class="result-grid"><div class="empty">Todavía no tienes clips generados.</div></div>
  <div id="zipArea"></div>
 </div>
</section>

<section class="page" id="projects">
 <div class="heading"><h1>Mis proyectos</h1><button class="btn small" onclick="navigate('create')">+ Nuevo</button></div>
 <div class="card"><h2>Biblioteca de SUKUNA</h2><div id="projectList" class="empty">Tus proyectos aparecerán aquí.</div></div>
</section>

<section class="page" id="captions">
 <h1>Editor de subtítulos</h1>
 <p class="muted">Personaliza el estilo de tus futuros clips.</p>
 <div class="card">
  <h2>Estilos disponibles</h2>
  <div class="grid">
   <div class="feature"><b>CLASSIC</b><p>Subtítulos blancos y limpios.</p><span class="tag">Clásico</span></div>
   <div class="feature"><b style="color:#f9e56b">BOLD</b><p>Subtítulos grandes y destacados.</p><span class="tag">Llamativo</span></div>
   <div class="feature"><b style="color:#bd7aff">NEON</b><p>Estética violeta brillante.</p><span class="tag">Neón</span></div>
   <div class="feature"><b>MINIMAL</b><p>Subtítulos sencillos.</p><span class="tag">Minimalista</span></div>
  </div>
 </div>
 <div class="card">
  <h2>Vista de ejemplo</h2>
  <div style="background:#08070d;padding:45px 15px;text-align:center;border-radius:12px">
   <span id="captionDemo" style="font-size:22px;font-weight:900;background:#8a57e9;padding:7px 12px;border-radius:5px">SUKUNA CLIPS</span>
  </div><br>
  <label>Cambiar texto de ejemplo</label>
  <input class="input" id="demoText" value="SUKUNA CLIPS" oninput="updateCaption()">
  <label>Color de texto</label>
  <input type="color" id="textColor" value="#ffffff" oninput="updateCaption()">
 </div>
</section>

<section class="page" id="settings">
 <h1>Configuración</h1>
 <p class="muted">Personaliza tu experiencia.</p>
 <div class="card">
  <h2>Perfil</h2>
  <label>Nombre del creador</label>
  <input class="input" id="creator" placeholder="Tu nombre">
  <button class="btn" onclick="saveSettings(this)">Guardar configuración</button>
 </div>
 <div class="card">
  <h2>Conexión del servidor</h2>
  <p class="muted">Dirección del backend de SUKUNA.</p>
  <input class="input" id="apiUrl" value="/api/process" placeholder="/api/process">
  <button class="btn" onclick="saveSettings(this)">Guardar servidor</button>
  <p class="muted" style="margin-top:12px">El servidor debe tener el endpoint de procesamiento configurado.</p>
 </div>
 <div class="card">
  <h2>Almacenamiento</h2>
  <button class="btn secondary" onclick="clearHistory()">Borrar historial local</button>
 </div>
</section>

<section class="page" id="plans">
 <h1>SUKUNA PRO</h1>
 <p class="muted">Planes previstos para la plataforma.</p>
 <div class="grid">
  <div class="card"><h2>FREE</h2><strong style="font-size:28px">$0</strong><p class="muted">Plan gratuito</p><br>
   <p class="muted">• Acceso básico</p><p class="muted">• Subida de videos</p><p class="muted">• Clips limitados</p><br>
   <span class="tag">Plan gratuito</span></div>
  <div class="card"><h2 class="purple">PRO</h2><strong style="font-size:28px">$9.99</strong><p class="muted">Precio mensual propuesto</p><br>
   <p class="muted">• Más procesamiento</p><p class="muted">• Más proyectos</p><p class="muted">• Herramientas avanzadas</p><p class="muted">• Exportaciones adicionales</p><br>
   <span class="tag">Próximamente</span></div>
 </div>
 <div class="notice">Los planes son demostrativos. No se realizan cobros ni se crean suscripciones hasta conectar un sistema de pagos.</div>
</section>

<section class="page" id="help">
 <h1>Centro de ayuda</h1>
 <div class="card"><h2>¿Cómo crear clips?</h2>
  <p class="muted">1. Selecciona un archivo autorizado.<br>2. Configura el formato y duración.<br>3. Confirma los permisos.<br>4. Pulsa generar clips.<br>5. Descarga los resultados.</p></div>
 <div class="card"><h2>¿Se pueden pegar enlaces de YouTube?</h2>
  <p class="muted">Sí. Puedes identificar y visualizar el video. Para procesarlo necesitas el archivo original o una fuente con autorización de uso.</p></div>
 <div class="card"><h2>Estado del proyecto</h2>
  <p class="muted">SUKUNA CLIPS V2: interfaz y conexión preparada para el motor de procesamiento.</p></div>
</section>

</main>
</div>
<div class="bottom-note">SUKUNA CLIPS V2 · AI VIDEO STUDIO · 2026</div>

<script>
function navigate(page){
 document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
 document.querySelectorAll(".nav").forEach(n=>n.classList.remove("active"));
 const s=document.getElementById(page); if(s)s.classList.add("active");
 document.querySelectorAll('[data-page="'+page+'"]').forEach(n=>n.classList.add("active"));
 window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll(".nav").forEach(i=>i.addEventListener("click",()=>navigate(i.dataset.page)));

function switchImport(type){
 const yt=type==="youtube";
 document.getElementById("youtubeBox").classList.toggle("hidden",!yt);
 document.getElementById("fileBox").classList.toggle("hidden",yt);
 document.getElementById("ytTab").classList.toggle("active",yt);
 document.getElementById("fileTab").classList.toggle("active",!yt);
}

function extractVideoId(value){
 value=value.trim();
 if(/^[a-zA-Z0-9_-]{11}$/.test(value))return value;
 try{
  const url=new URL(value);let id="";
  if(url.hostname==="youtu.be"){id=url.pathname.split("/")[1]||"";}
  else if(["youtube.com","www.youtube.com","m.youtube.com","music.youtube.com"].includes(url.hostname)){
   id=url.searchParams.get("v")||"";
   if(!id&&(url.pathname.startsWith("/shorts/")||url.pathname.startsWith("/embed/")))id=url.pathname.split("/")[2]||"";
  }
  return /^[a-zA-Z0-9_-]{11}$/.test(id)?id:"";
 }catch(e){return "";}
}

function previewYoutube(){
 const id=extractVideoId(document.getElementById("youtubeInput").value);
 const container=document.getElementById("youtubeResult");
 container.replaceChildren();
 if(!id){container.textContent="Introduce un enlace o código válido.";return;}

 const iframe=document.createElement("iframe");
 iframe.id="youtubeFrame";
 iframe.src="https://www.youtube-nocookie.com/embed/"+id;
 iframe.title="Vista previa del video";
 iframe.referrerPolicy="strict-origin-when-cross-origin";
 iframe.allow="accelerometer; autoplay; encrypted-media; picture-in-picture; fullscreen";
 iframe.allowFullscreen=true;

 const info=document.createElement("p");
 info.className="muted";
 info.textContent="Video identificado: "+id;
 container.append(iframe,info);
}

let selectedFile=null,localUrl=null;
document.getElementById("videoInput").addEventListener("change",function(){
 const file=this.files[0]; if(!file)return;
 selectedFile=file;
 if(localUrl)URL.revokeObjectURL(localUrl);
 localUrl=URL.createObjectURL(file);
 const p=document.getElementById("localPreview");
 p.src=localUrl;p.classList.remove("hidden");
 document.getElementById("fileInfo").textContent="Archivo: "+file.name+" · "+(file.size/1024/1024).toFixed(2)+" MB";
});

async function generateClips(){
 const status=document.getElementById("status"),button=document.getElementById("generate"),results=document.getElementById("results"),file=selectedFile;
 if(!file){status.textContent="Primero selecciona un archivo de video.";return;}
 if(!document.getElementById("rights").checked){status.textContent="Confirma que tienes autorización para procesar el video.";return;}
 if(file.size>500*1024*1024){status.textContent="El archivo supera el límite inicial de 500 MB.";return;}
 button.disabled=true;results.replaceChildren();
 const bar=document.getElementById("progressBar");
 bar.style.width="10%";status.textContent="Enviando video a SUKUNA AI...";
 const form=new FormData();form.append("file",file);
 form.append("ratio",document.getElementById("ratio").value);
 form.append("clips",document.getElementById("clipAmount").value);
 form.append("duration",document.getElementById("duration").value);
 form.append("captions",document.getElementById("captions").checked);
 form.append("style",document.getElementById("captionStyle").value);
 try{
  const endpoint=document.getElementById("apiUrl").value.trim();
  if(!endpoint)throw new Error("Configura la dirección del servidor.");
  bar.style.width="35%";
  const response=await fetch(endpoint,{method:"POST",body:form});
  bar.style.width="70%";
  const data=await response.json();
  if(!response.ok)throw new Error(data.detail||"Error en el procesamiento.");
  if(!data.clips||!data.clips.length)throw new Error("El servidor no devolvió clips.");
  data.clips.forEach((clip,i)=>{
   const card=document.createElement("div");card.className="clip";
   const t=document.createElement("h3");t.textContent="✂ Clip "+(i+1);
   const s=document.createElement("div");s.className="score";s.textContent="Viral Score estimado: "+clip.score+"/100";
   const d=document.createElement("p");d.textContent=clip.text||"Clip generado.";
   const tm=document.createElement("p");tm.textContent="Inicio: "+clip.start+"s · Final: "+clip.end+"s";
   const sb=document.createElement("p");sb.textContent="Subtítulos: "+clip.subtitle;
   card.append(t,s,d,tm,sb);results.appendChild(card);
  });
  const link=document.createElement("a");
  link.className="btn";link.style.cssText="display:inline-block;margin-top:20px;text-decoration:none";
  link.href=data.download;link.textContent="⬇ Descargar todos los clips";
  document.getElementById("zipArea").replaceChildren(link);
  saveProject({name:file.name,date:new Date().toLocaleString(),count:data.clips.length});
  bar.style.width="100%";
  status.textContent="¡SUKUNA terminó! Se generaron "+data.clips.length+" clips.";
  updateStats();
 }catch(error){
  bar.style.width="0%";
  status.textContent="No se pudo procesar: "+error.message+". Comprueba que el backend esté encendido.";
 }finally{button.disabled=false;}
}

function getProjects(){
 try{return JSON.parse(localStorage.getItem("sukunaProjects")||"[]");}catch(e){return [];}
}
function saveProject(project){
 const projects=getProjects();projects.unshift(project);
 try{localStorage.setItem("sukunaProjects",JSON.stringify(projects.slice(0,50)));}catch(e){}
}
function renderProjects(){
 const projects=getProjects(),list=document.getElementById("projectList");
 list.replaceChildren();
 if(!projects.length){list.textContent="Tus proyectos aparecerán aquí.";return;}
 projects.forEach(p=>{
  const item=document.createElement("div");item.className="feature";item.style.marginBottom="10px";
  const b=document.createElement("b");b.textContent=p.name;
  const d=document.createElement("p");d.className="muted";d.textContent=p.date+" · "+p.count+" clips";
  item.append(b,d);list.appendChild(item);
 });
}
function updateStats(){
 const projects=getProjects();
 document.getElementById("projectCount").textContent=projects.length;
 document.getElementById("clipCount").textContent=projects.reduce((s,p)=>s+(p.count||0),0);
 renderProjects();
}
function clearHistory(){
 try{localStorage.removeItem("sukunaProjects");}catch(e){}
 updateStats();
}
function updateCaption(){
 const demo=document.getElementById("captionDemo");
 demo.textContent=document.getElementById("demoText").value;
 demo.style.color=document.getElementById("textColor").value;
}
function saveSettings(btn){
 try{
  localStorage.setItem("sukunaCreator",document.getElementById("creator").value);
  localStorage.setItem("sukunaApi",document.getElementById("apiUrl").value);
 }catch(e){}
 if(btn){const old=btn.textContent;btn.textContent="✓ Guardado";setTimeout(()=>btn.textContent=old,1500);}
}
function loadSettings(){
 try{
  document.getElementById("creator").value=localStorage.getItem("sukunaCreator")||"";
  document.getElementById("apiUrl").value=localStorage.getItem("sukunaApi")||"/api/process";
 }catch(e){}
}
loadSettings();updateStats();
</script>
</body>
</html>
'''
