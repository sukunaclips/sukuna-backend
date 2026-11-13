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
from fastapi.responses import FileResponse
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
    """Devuelve [(inicio, fin, texto)]. Si no hay Whisper, devuelve []."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, compute_type="int8")
    segments, _ = _model.transcribe(str(path), vad_filter=True)
    return [(s.start, s.end, s.text.strip()) for s in segments]


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
        return "crop='min(iw,ih*9/16)':ih,scale=1080:1920"
    if ratio == "1:1":
        return "crop='min(iw,ih)':'min(iw,ih)',scale=1080:1080"
    return ("scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2")


def cut_clip(src, clip, out, ratio, srt, style):
    vf = ratio_filter(ratio)
    cwd = None
    if srt and srt.exists() and srt.stat().st_size:
        vf += f",subtitles={srt.name}:force_style='{STYLES.get(style, STYLES['classic'])}'"
        cwd = srt.parent
    run(["ffmpeg", "-y", "-ss", str(clip["start"]), "-i", str(src.resolve()),
         "-t", str(clip["end"] - clip["start"]), "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
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
        segs = transcribe(src)
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


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html")
