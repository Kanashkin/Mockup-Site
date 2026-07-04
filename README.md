# T-Shirt Mockup Site

Generates realistic t-shirt mockups using exact Photoshop warp data.

## Structure

```
mockup_site/
├── server.py            ← FastAPI server
├── requirements.txt
├── static/
│   └── index.html       ← Frontend
└── mockup_package/      ← Extracted from PSD (copy here)
    ├── mockup.json
    ├── shirt_base.png
    ├── tshirt_mask.png
    ├── overlay_Grey.png
    ├── overlay_Light.png
    ├── overlay_Shadow.png
    └── overlay_Texture.png
```

## Local run

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
# → open http://localhost:8000
```

## Deploy on Railway (free)

1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Done — Railway gives you a public URL

## Deploy on Render (free)

1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

## Add more mockups

Run `extract_mockup.py` on any PSD to generate a new `mockup_package/`,
then swap it in. Each mockup needs its own server instance, or modify
`server.py` to support multiple mockups with a selector on the frontend.

## API

```
POST /render
  Content-Type: multipart/form-data
  file: <image file>
  → Returns PNG image
```
