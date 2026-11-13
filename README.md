# SUKUNA backend

## Local
1. Instala ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`)
2. `pip install -r requirements.txt`
3. `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Prueba: http://localhost:8000/api/health

## Online (Render / Railway / Fly.io)
Sube esta carpeta a GitHub y despliega con el Dockerfile.
Copia la URL pública y pégala en la página, en
Configuración > Conexión del servidor:  https://TU-APP/api/process

## Variables opcionales
WHISPER_MODEL=tiny|base|small   MAX_UPLOAD_MB=500   ALLOWED_ORIGINS=https://tu-sitio
