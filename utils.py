import os
import json
import faiss
import torch
import yt_dlp
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Enable debugging prints if needed
debug_mode = True
def debug_print(message, level=1):
    if debug_mode:
        prefix = "  " * level
        print(f"{prefix}🔹 {message}")

# CONSTANTS: File paths and folders (adjust if your files are in other locations)
IMAGE_DIR = "images"
FIGURES_JSON = "output.json"
KNOWLEDGEBASE_JSON = "knowledgebase.json"
METADATA_JSON = "metadata.json"
FAISS_TEXT_INDEX = "textbook_faiss.index"
FAISS_FIGURES_INDEX = "subchapter_faiss.index"
METADATA_FIGURES_JSON = "subchapter_metadata.json"

# Load Knowledge Base and metadata JSON files
with open(KNOWLEDGEBASE_JSON, "r", encoding="utf-8") as f:
    kb_data = json.load(f)
with open(METADATA_JSON, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Normalize titles helper for consistent searching
def normalize_title(title):
    return title.strip().lower()

# Build a dict for quick content retrieval, keys are (chapter, normalized_title)
normalized_kb = {}
for chapter, topics in kb_data.items():
    for title, content in topics.items():
        norm_key = (chapter, normalize_title(title))
        normalized_kb[norm_key] = content

# Set device for model (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SentenceTransformer model for embeddings and semantic search
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.to(device)

# Load FAISS index for textbook content
faiss_index = faiss.read_index(FAISS_TEXT_INDEX)

# Main search function implementing hybrid exact + semantic search
def search(query, top_k=5, similarity_threshold=0.98, mode="hybrid"):
    norm_query = normalize_title(query)
    results = []
    seen_embeddings = []
    seen_titles = set()

    # Exact match search helper
    def get_exact_matches():
        for item in metadata:
            title = item["title"]
            chapter = item["chapter"]
            norm_title = normalize_title(title)
            if norm_query in norm_title:
                norm_key = (chapter, norm_title)
                content = normalized_kb.get(norm_key)
                if content:
                    seen_titles.add(norm_key)
                    return [{
                        "title_key": title,
                        "chapter": chapter,
                        "score": 0.0,
                        "content": content
                    }]
        return []

    # Semantic match search helper
    def get_semantic_matches():
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        semantic_results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            raw_title = metadata[idx]["title"]
            chapter = metadata[idx]["chapter"]
            norm_key = (chapter, normalize_title(raw_title))
            content = normalized_kb.get(norm_key)
            if content and norm_key not in seen_titles:
                content_embedding = model.encode(content, convert_to_tensor=True)
                is_duplicate = False
                for prev_emb in seen_embeddings:
                    if util.cos_sim(content_embedding, prev_emb).item() >= similarity_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    seen_embeddings.append(content_embedding)
                    seen_titles.add(norm_key)
                    semantic_results.append({
                        "title_key": raw_title,
                        "chapter": chapter,
                        "score": distances[0][i],
                        "content": content
                    })
        return semantic_results

    if mode == "exact":
        results = get_exact_matches()
    elif mode == "semantic":
        results = get_semantic_matches()
    else:  # hybrid
        results = get_exact_matches()
        if not results:
            results = get_semantic_matches()
    return results

# Load figures data and metadata for image retrieval
with open(FIGURES_JSON, "r", encoding="utf-8") as f:
    figures_data = json.load(f)

# Separate model and FAISS index for figures/subchapter search
image_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
fig_faiss_index = faiss.read_index(FAISS_FIGURES_INDEX)

with open(METADATA_FIGURES_JSON, "r", encoding="utf-8") as f:
    metadata_figures = json.load(f)

# Search exact figure subchapter helper
def search_exact_subchapter(query, top_k=1):
    query_embedding = image_model.encode([query], convert_to_numpy=True).astype('float32').reshape(1, -1)
    _, indices = fig_faiss_index.search(query_embedding, top_k)
    best_index = str(indices[0][0])
    return metadata_figures.get(best_index, None)

# Retrieve local image path for a figure, trying common extensions and patterns
def get_image_path(figure_ref):
    base_name = figure_ref.replace(" ", "_")
    attempts = [f"{base_name}.png", f"{base_name}.jpg", f"figure_{base_name}.png"]
    for attempt in attempts:
        test_path = os.path.join(IMAGE_DIR, attempt)
        if os.path.exists(test_path):
            return test_path
    return None

# Fetch only figures metadata + path for a given subchapter name
def fetch_figures_only(subchapter_name):
    figures = [fig for fig in figures_data if fig["subchapter"] == subchapter_name]
    if not figures:
        return "No relevant figures found."
    figure_blocks = []
    for fig in figures:
        img_path = get_image_path(fig['figure'])
        if img_path:
            figure_blocks.append({
                "name": fig['figure'],
                "path": img_path,
                "desc": fig['description']
            })
    return figure_blocks

# Helper to retrieve figures and generate simple HTML for rendering (if needed)
def retrieve_and_expand_figures(query):
    search_results = search(query, mode="hybrid", top_k=1)
    if not search_results:
        return "<p>No relevant text found for image retrieval.</p>"
    best_text_match = search_results[0]
    subchapter_name = best_text_match["title_key"]
    blocks = fetch_figures_only(subchapter_name)
    if isinstance(blocks, str):
        return f"<p>{blocks}</p>"
    figure_html = "<div style='margin-top: 20px;'><h3>📊 Visual Aids</h3>"
    for fig in blocks[:3]:
        clean_desc = fig['desc']
        figure_html += f"""
        <div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
            <img src='{fig['path']}' style='max-width: 100%; height: auto; display: block; margin: 0 auto;'>
            <p style='text-align: center; font-style: italic;'>{clean_desc or 'Visual demonstration'}</p>
        </div>
        """
    figure_html += "</div>"
    return figure_html

# Fetch animated explainer videos from YouTube via yt-dlp
def fetch_animated_videos(topic, num_videos=1):
    search_query = f"ytsearch{num_videos}:{topic} animation explained in english"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=False)
        if "entries" in info and len(info["entries"]) > 0:
            video = info["entries"][0]
            if video.get("duration", 301) <= 300:  # Max 5 minutes duration
                return {
                    "title": video["title"],
                    "url": video["url"],
                    "id": video["id"]
                }
    return None
