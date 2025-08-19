import streamlit as st
import time
import google.generativeai as genai
from youtubesearchpython import VideosSearch
import yt_dlp

ROBOT_IMG = "robot.png"
LESSON_IMG = "lesson.png"

# Configure Gemini API
genai.configure(api_key="API KEY")

def generate_lesson_from_gemini(topic):
    prompt = f"""
    You are an AI teacher. Explain the topic "{topic}" in a detailed, student-friendly way.

    ✅ Instructions:
    1. Write each sentence on a new line.
    2. Insert the marker `<<image:start>>` after a few sentences.
    3. After that, write 7-8 sentences explaining with reference to the image.
    4. Insert `<<image:end>>` when done with the image explanation.
    5. Continue normal explanation after.
    6. At the END, insert the marker `<<video>>` for the related YouTube video.
    7. Only output explanation + markers, one per line.
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    content = response.text.strip()
    return content.split("\n")

def fetch_youtube_video(topic):
    try:
        search = VideosSearch(topic, limit=1)
        results = search.result()
        if results and "result" in results and len(results["result"]) > 0:
            video_id = results["result"][0]["id"]
            return f"https://www.youtube.com/embed/{video_id}?autoplay=1"
    except Exception as e:
        print("YouTube search failed:", e)

    # 🔥 Fallback with yt-dlp
    try:
        query_url = f"ytsearch1:{topic}"
        ydl_opts = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query_url, download=False)
            if "entries" in info and len(info["entries"]) > 0:
                video_id = info["entries"][0]["id"]
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1"
    except Exception as e:
        print("yt-dlp fallback failed:", e)

    return None

def run_lesson(sentences, video_url):
    st.title("AI Teacher Robot")
    placeholder = st.empty()
    display_robot = True
    placeholder.image(ROBOT_IMG, caption="Robot Face", use_container_width=True)

    for sentence in sentences:
        if sentence.strip() == "<<image:start>>":
            display_robot = False
            placeholder.image(LESSON_IMG, caption="Lesson Image", use_container_width=True)
            continue
        elif sentence.strip() == "<<image:end>>":
            display_robot = True
            placeholder.image(ROBOT_IMG, caption="Robot Face", use_container_width=True)
            continue
        elif sentence.strip() == "<<video>>":
            if video_url:
                st.markdown(
                    f"""
                    <iframe width="560" height="315"
                    src="{video_url}"
                    frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
                    """,
                    unsafe_allow_html=True
                )
            continue

        if display_robot:
            placeholder.image(ROBOT_IMG, caption="Robot Face", use_container_width=True)
        else:
            placeholder.image(LESSON_IMG, caption="Lesson Image", use_container_width=True)

        with st.empty():
            st.write(sentence)
        time.sleep(2)

if __name__ == "__main__":
    topic = st.text_input("Enter lesson topic:", "Renewable Energy")
    if st.button("Start Lesson"):
        sentences = generate_lesson_from_gemini(topic)
        video_url = fetch_youtube_video(topic)
        run_lesson(sentences, video_url)
