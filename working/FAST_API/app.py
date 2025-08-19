# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
import yt_dlp

app = FastAPI()

# Mount static folder for images, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder for HTML
templates = Jinja2Templates(directory="templates")

# Configure Gemini API
genai.configure(api_key="AIzaSyDcr5um0rWBPVk5Y7B0twoES_U3bghOLns")

def generate_lesson_from_gemini(topic):
    prompt = f"""
    You are an AI teacher. Explain the topic "{topic}" in a detailed, student-friendly way.
    ✅ Instructions:
    1. Each sentence on a new line.
    2. Insert `<<image:start>>` after a few sentences.
    3. Write 7–8 sentences about the image.
    4. Insert `<<image:end>>`.
    5. At the END insert `<<video>>`.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip().split("\n")

def fetch_youtube_video(topic):
    query_url = f"ytsearch1:{topic}"
    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_url, download=False)
        if "entries" in info and len(info["entries"]) > 0:
            video_id = info["entries"][0]["id"]
            return f"https://www.youtube.com/embed/{video_id}?autoplay=1"
    return None

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/lesson")
async def lesson(topic: str):
    sentences = generate_lesson_from_gemini(topic)
    video_url = fetch_youtube_video(topic)
    return JSONResponse({"sentences": sentences, "video_url": video_url})
