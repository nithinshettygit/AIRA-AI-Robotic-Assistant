"""
Robust AI Tutor - single file (no Google Drive)

Key features:
1) Student can start a lesson ("start lesson on X" / "teach me about X" / "lesson on X")
   or just chat; router auto-selects the right agent.
2) Two-layer memory:
   - Short-term: in-state conversation buffer
   - Long-term: FAISS vectorstore (saves useful snippets; retrieved for RAG & doubts)
3) In-lesson: always includes (a) diagram availability line and (b) YouTube link line.
   If image file is missing, we still tell the student how to request it.
4) Doubt resolving (interrupt) works regardless of lesson state; if in-lesson, it
   restores lesson context after answering.
5) Embeddings: thenlper/gte-large via HuggingFaceBgeEmbeddings (no InstructorEmbedding).
6) Uses Groq-compatible ChatOpenAI via base_url + api key (env GROQ_API_KEY recommended).

Env vars you can set:
- GROQ_API_KEY     (or hardcode API_KEY below for quick tests)
- LLM_MODEL        default: llama3-8b-8192
- LLM_BASE_URL     default: https://api.groq.com/openai/v1
- DATA_PATH        default: data/
- EMBEDDING_MODEL  default: thenlper/gte-large
"""

import os
import re
import json
import uuid
import time
import traceback
from typing import List, Dict, Any, Optional, TypedDict

# ---- External deps ----
try:
    import torch
    from dotenv import load_dotenv
    from IPython.display import Image, display, Markdown

    # LangChain (community & openai shims)
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_openai import ChatOpenAI

    # Tools
    try:
        from langchain.tools import YouTubeSearchTool
        HAVE_YT_TOOL = True
    except Exception:
        HAVE_YT_TOOL = False
        try:
            from youtube_search import YoutubeSearch
        except Exception:
            YoutubeSearch = None

    # LangGraph
    from langgraph.graph import StateGraph, END

except Exception as e:
    print("Import error: install packages: langchain, langchain-community, langchain-openai, langgraph, faiss-cpu, sentence-transformers, python-dotenv, torch")
    raise

# ---- Load env ----
load_dotenv()

# ---- Config ----
DATA_PATH = os.getenv("DATA_PATH", "data/")
VECTOR_DB_DIR = os.path.join(DATA_PATH, "faiss_vectorstore_final")
CONTENT_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "content_faiss_index")
SUBCHAPTER_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "subchapter_faiss_index")
MEMORY_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "memory_faiss_index")

MERGED_CHUNKS_FILE = os.path.join(DATA_PATH, "merged_chunks_with_figures.json")
SUBCHAPTER_METADATA_FILE = os.path.join(DATA_PATH, "subchapter_metadata.json")
IMAGE_BASE_PATH = os.path.join(DATA_PATH, "images")

API_KEY =  ""
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Embeddings ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

# ---- Vectorstore helpers ----
def _load_faiss(path: str, embeddings) -> Optional[FAISS]:
    try:
        if os.path.exists(path):
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load FAISS at {path}: {e}")
    return None

def _save_faiss(vs: FAISS, path: str):
    try:
        os.makedirs(path, exist_ok=True)
        vs.save_local(path)
    except Exception as e:
        print("Failed to save FAISS:", e)

# ---- Load content/subchapter indexes (optional if you don't have them yet) ----
content_db = _load_faiss(CONTENT_INDEX_PATH, embeddings)
subchapter_db = _load_faiss(SUBCHAPTER_INDEX_PATH, embeddings)
memory_db = _load_faiss(MEMORY_INDEX_PATH, embeddings)  # may be None initially

# ---- Load JSONs (optional) ----
FIGURE_LOOKUP: Dict[str, Dict[str, Any]] = {}
CONTENT_CHUNKS: List[Dict[str, Any]] = []
SUBCHAPTER_MAP: Dict[str, Any] = {}

if os.path.exists(MERGED_CHUNKS_FILE):
    try:
        with open(MERGED_CHUNKS_FILE, "r", encoding="utf-8") as f:
            CONTENT_CHUNKS = json.load(f)
        for c in CONTENT_CHUNKS:
            figs = c.get("figures") or []
            for fig in figs:
                name = fig.get("figure") or fig.get("name")
                if name:
                    FIGURE_LOOKUP[name] = {
                        "desc": fig.get("desc", ""),
                        "file": fig.get("file", name.replace(" ", "_") + ".png")
                    }
    except Exception as e:
        print("Failed to load merged chunks:", e)

if os.path.exists(SUBCHAPTER_METADATA_FILE):
    try:
        with open(SUBCHAPTER_METADATA_FILE, "r", encoding="utf-8") as f:
            SUBCHAPTER_MAP = json.load(f)
    except Exception as e:
        print("Failed to load subchapter metadata:", e)

# ---- LLM ----
if not API_KEY:
    print("WARNING: GROQ_API_KEY not set; set it in env for real calls.")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    temperature=0.2,
)

def call_llm(system: str, user: str) -> str:
    """Robust single-call helper."""
    try:
        # Use the Chat API with explicit roles for reliability
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        print("LLM call failed:", e)
        traceback.print_exc()
        return "[LLM failure]"

# ---- Memory store (long-term via FAISS) ----
class MemoryStore:
    def __init__(self, embeddings, path=MEMORY_INDEX_PATH):
        self.embeddings = embeddings
        self.path = path
        self.vs: Optional[FAISS] = _load_faiss(path, embeddings)

    def add(self, text: str, metadata: Dict[str, Any]):
        metadata = dict(metadata)
        metadata.setdefault("timestamp", time.time())
        metadata.setdefault("id", str(uuid.uuid4()))
        from langchain_core.documents import Document
        doc = Document(page_content=text, metadata=metadata)
        if self.vs is None:
            try:
                self.vs = FAISS.from_documents([doc], self.embeddings)
                _save_faiss(self.vs, self.path)
            except Exception as e:
                print("Failed creating memory vs:", e)
        else:
            try:
                self.vs.add_documents([doc])
                _save_faiss(self.vs, self.path)
            except Exception as e:
                print("Failed adding memory doc:", e)

    def retrieve(self, query: str, k: int = 4):
        if self.vs is None:
            return []
        try:
            return self.vs.similarity_search(query, k=k)
        except Exception as e:
            print("Memory retrieve error:", e)
            return []

memory_store = MemoryStore(embeddings)

# ---- Utilities ----
def extract_topic_from_query(q: str) -> str:
    patterns = [
        r'start lesson on (.+)$',
        r'begin lesson on (.+)$',
        r'teach me about (.+)$',
        r'lesson on (.+)$',
        r'^\s*lesson\s+(.+)$',
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

def retrieve_content(query: str, k: int = 4):
    if content_db is None:
        return []
    try:
        return content_db.similarity_search(query, k=k)
    except Exception as e:
        print("Content retrieval error:", e)
        return []

def is_query_in_syllabus(query: str) -> bool:
    if subchapter_db is None:
        return False
    try:
        return len(subchapter_db.similarity_search(query, k=1)) > 0
    except Exception:
        return False

def get_figure_path(figure_name: str) -> str:
    info = FIGURE_LOOKUP.get(figure_name)
    filename = (info.get('file') if info else (figure_name.replace(" ", "_") + ".png"))
    return os.path.join(IMAGE_BASE_PATH, filename)

def find_figure_for_topic(topic: str) -> Optional[str]:
    if not topic:
        return None
    # exact match by key
    for name in FIGURE_LOOKUP.keys():
        if topic.lower() == name.lower():
            return name
    # substring match
    for name, v in FIGURE_LOOKUP.items():
        desc = v.get("desc", "")
        if topic.lower() in name.lower() or topic.lower() in desc.lower():
            return name
    # heuristic: look for "Figure X.Y" in content chunks related to topic
    pat = re.compile(r'Figure\s*\d+(\.\d+)?', re.IGNORECASE)
    for c in CONTENT_CHUNKS:
        if topic.lower() in (c.get("text") or "").lower():
            figs = pat.findall(c.get("text", ""))
            if figs:
                return figs[0]
    return None

def get_youtube_for_topic(topic: str) -> Optional[str]:
    topic = (topic or "").strip()
    if not topic:
        return None
    if HAVE_YT_TOOL:
        try:
            yt = YouTubeSearchTool()
            out = yt.run(f"{topic} educational tutorial")
            for line in (out or "").splitlines():
                if line.strip().startswith("http"):
                    return line.strip()
        except Exception:
            return None
    else:
        if YoutubeSearch:
            try:
                results = YoutubeSearch(topic + " educational tutorial", max_results=5).to_dict()
                if results:
                    video_id = results[0].get("id") or results[0].get("url_suffix","").replace("/watch?v=","")
                    if video_id:
                        return "https://www.youtube.com/watch?v=" + video_id
            except Exception:
                return None
    return None

# ---- State ----
class GraphState(TypedDict):
    history: List[List[str]]          # [["user","..."], ["ai","..."], ...]
    user_query: str
    next: str
    relevant_docs: Optional[list]
    figure_name: Optional[str]
    figure_path: Optional[str]
    youtube_url: Optional[str]
    current_topic: Optional[str]
    lesson_progress: int
    in_lesson: bool
    pre_interrupt_state: Optional[Dict[str, Any]]
    topics_covered: List[str]

def initialize_state() -> GraphState:
    return {
        "history": [],
        "user_query": "",
        "next": "teacher",
        "relevant_docs": None,
        "figure_name": None,
        "figure_path": None,
        "youtube_url": None,
        "current_topic": None,
        "lesson_progress": 0,
        "in_lesson": False,
        "pre_interrupt_state": None,
        "topics_covered": [],
    }

# ---- Router ----
def route_query(state: GraphState) -> GraphState:
    # support tests that set state['messages']
    if 'messages' in state and state['messages']:
        last = state['messages'][-1]
        if isinstance(last, tuple) and len(last) == 2:
            state['user_query'] = last[1]
        else:
            state['user_query'] = str(last)

    q = (state['user_query'] or "").strip()

    # Interrupts ALWAYS route to interrupt (even if not in a lesson)
    m = re.match(r'^(doubt|question|clarify)[:\-\s]*(.+)', q, re.IGNORECASE)
    if m:
        if state.get("in_lesson"):
            state["pre_interrupt_state"] = {
                "current_topic": state.get("current_topic"),
                "lesson_progress": state.get("lesson_progress", 0),
            }
        state["user_query"] = m.group(2).strip()
        state["next"] = "interrupt"
        return state

    # Quick topic summary intent
    if re.search(r'(what have we (discussed|covered))|(topics covered)|(which topics)', q, re.IGNORECASE):
        state["next"] = "summary"
        return state

    # Lesson start
    topic = extract_topic_from_query(q)
    if topic:
        state["current_topic"] = topic
        state["in_lesson"] = True
        state["lesson_progress"] = 0
        if topic not in state["topics_covered"]:
            state["topics_covered"].append(topic)
        state["next"] = "rag"
        return state

    # Media intents
    low = q.lower()
    if any(kw in low for kw in ["show diagram", "show figure", "diagram", "show image"]):
        state["next"] = "image"
        return state
    if "show video" in low or "video" in low:
        state["next"] = "youtube"
        return state

    # If in lesson, continue lesson; else general
    state["next"] = "rag" if state.get("in_lesson") else "general"
    return state

# ---- Agents ----
def rag_agent(state: GraphState) -> GraphState:
    user_q = state["user_query"]
    topic = state.get("current_topic") or user_q

    # Retrieval
    if state.get("in_lesson") and state.get("lesson_progress", 0) == 0:
        docs = retrieve_content(topic, k=4)
    else:
        docs = retrieve_content(user_q, k=4)
    state["relevant_docs"] = docs

    # Memory recall
    mems = memory_store.retrieve(user_q, k=3)
    context_parts: List[str] = []
    for d in docs or []:
        try:
            context_parts.append(d.page_content)
        except Exception:
            pass
    for m in mems or []:
        try:
            context_parts.append(f"(memory) {m.page_content}")
        except Exception:
            pass
    context = "\n\n".join(context_parts) if context_parts else ""

    # Compose lesson / answer
    if state.get("in_lesson") and state.get("lesson_progress", 0) == 0:
        system = (
            "You are an expert teacher. Write a friendly, concise lesson intro and 2–4 learning objectives. "
            "Then give a short first explanation (6–10 lines). Keep it clear and non-repetitive."
        )
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Context (RAG + memory):\n{context}\n\n"
            "Output sections:\n"
            "1) **Lesson Introduction**\n"
            "2) **Learning Objectives** (bullets)\n"
            "3) **First Explanation**\n"
            "4) **Resources**: two lines:\n"
            "   - Diagram: write 'diagram available: <name or say none>. Say: ask “show diagram”'\n"
            "   - Video: paste one YouTube URL if found, else say 'no suitable video found'\n"
        )
    else:
        system = (
            "You are a helpful tutor continuing a lesson or answering a question. "
            "Use provided context. Be concise and suggest one next step."
        )
        user_prompt = (
            f"Current topic (if any): {topic}\n"
            f"Student query: {user_q}\n\n"
            f"Context (RAG + memory):\n{context}\n\n"
            "Answer clearly in <=10 lines, then add 'Next step:' one actionable suggestion."
        )

    answer = call_llm(system, user_prompt)

    # Always enrich with diagram/video lines if starting a lesson
    if state.get("in_lesson") and state.get("lesson_progress", 0) == 0:
        available_figure = find_figure_for_topic(topic) if topic else None
        if available_figure:
            fig_path = get_figure_path(available_figure)
            fig_line = f"Diagram available: {available_figure} (ask: 'show diagram')"
            if os.path.exists(fig_path):
                # (Optional) show inline if running in notebook
                try:
                    display(Image(filename=fig_path))
                except Exception:
                    pass
        else:
            fig_line = "Diagram available: none"

        yt_link = get_youtube_for_topic(topic)
        yt_line = f"Video: {yt_link}" if yt_link else "Video: no suitable video found"

        # if not already included by the model, add at the end for determinism
        answer += f"\n\n**Resources**\n- {fig_line}\n- {yt_line}"

    # history + long-term memory
    state["history"].append(["ai", answer])
    memory_store.add(answer, {"type": "lesson" if state.get("in_lesson") else "answer",
                              "topic": topic or "",
                              "source": "rag_agent"})

    if state.get("in_lesson"):
        state["lesson_progress"] = state.get("lesson_progress", 0) + 1

    state["next"] = "end"
    return state

def image_agent(state: GraphState) -> GraphState:
    q = state["user_query"]
    topic = state.get("current_topic", "")
    # specific figure?
    m = re.search(r'(figure\s*\d+(\.\d+)?)', q, re.IGNORECASE)
    figure_name = m.group(1) if m else None
    if not figure_name and topic:
        figure_name = find_figure_for_topic(topic)

    if figure_name:
        path = get_figure_path(figure_name)
        desc = (FIGURE_LOOKUP.get(figure_name) or {}).get("desc", "")
        state["figure_name"] = figure_name
        state["figure_path"] = path
        if os.path.exists(path):
            try:
                display(Image(filename=path))
                display(Markdown(f"**Figure:** {figure_name}\n\n**Description:** {desc or '—'}"))
                state["history"].append(["ai", f"Displayed figure {figure_name}: {desc or '—'}"])
            except Exception:
                state["history"].append(["ai", f"Diagram {figure_name} available (file present). Description: {desc or '—'}"])
        else:
            state["history"].append(["ai", f"Diagram '{figure_name}' is available but file is not stored locally. Description: {desc or '—'}"])
    else:
        state["history"].append(["ai", "No diagram mapped for this topic. You can continue with text or ask for a video."])

    state["next"] = "end"
    return state

def youtube_agent(state: GraphState) -> GraphState:
    topic = state.get("current_topic") or state["user_query"]
    url = get_youtube_for_topic(topic)
    if url:
        state["youtube_url"] = url
        try:
            display(Markdown(f"**Video for _{topic}_**: {url}"))
        except Exception:
            pass
        state["history"].append(["ai", f"Video: {url}"])
    else:
        state["history"].append(["ai", "No suitable video found. Would you like a concise text explanation?" ])
    state["next"] = "end"
    return state

def general_agent(state: GraphState) -> GraphState:
    q = state["user_query"]
    system = "You are a friendly tutor. Answer conversationally and concisely."
    answer = call_llm(system, q)
    state["history"].append(["ai", answer])
    memory_store.add(answer, {"type": "qa", "source": "general_agent"})
    state["next"] = "end"
    return state

def interrupt_agent(state: GraphState) -> GraphState:
    q = state["user_query"]
    docs = retrieve_content(q, k=3)
    mems = memory_store.retrieve(q, k=3)

    context_parts: List[str] = []
    for d in docs or []:
        try: context_parts.append(d.page_content)
        except: pass
    for m in mems or []:
        try: context_parts.append(f"(memory) {m.page_content}")
        except: pass
    context = "\n\n".join(context_parts) if context_parts else ""

    system = "You are a concise tutor answering a doubt. Use context if helpful."
    user_prompt = f"Question: {q}\n\nContext:\n{context}\n\nAnswer briefly and clearly."

    answer = call_llm(system, user_prompt)
    state["history"].append(["ai", answer])
    memory_store.add(answer, {"type": "doubt_answer", "source": "interrupt"})

    pre = state.get("pre_interrupt_state") or {}
    if pre:
        state["current_topic"] = pre.get("current_topic", state.get("current_topic"))
        state["lesson_progress"] = pre.get("lesson_progress", state.get("lesson_progress"))
        state["pre_interrupt_state"] = None

    state["next"] = "end"
    return state

def summary_agent(state: GraphState) -> GraphState:
    topics = state.get("topics_covered", [])
    if topics:
        msg = "So far we've covered: " + ", ".join(topics)
    else:
        msg = "We haven't covered any topics yet."
    state["history"].append(["ai", msg])
    state["next"] = "end"
    return state

# ---- Build graph ----
def create_graph():
    builder = StateGraph(GraphState)
    builder.add_node("teacher", route_query)
    builder.add_node("rag", rag_agent)
    builder.add_node("general", general_agent)
    builder.add_node("image", image_agent)
    builder.add_node("youtube", youtube_agent)
    builder.add_node("interrupt", interrupt_agent)
    builder.add_node("summary", summary_agent)
    builder.set_entry_point("teacher")

    builder.add_conditional_edges(
        "teacher",
        lambda s: s.get("next", "general"),
        {
            "rag": "rag",
            "general": "general",
            "image": "image",
            "youtube": "youtube",
            "interrupt": "interrupt",
            "summary": "summary",
        }
    )
    for n in ["rag", "general", "image", "youtube", "interrupt", "summary"]:
        builder.add_edge(n, END)
    return builder.compile()

graph = create_graph()

# ---- REPL ----
def chat_with_tutor():
    print("Welcome to the Robust AI Tutor (type 'exit' to quit)")
    state = initialize_state()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            user_input = input("\nStudent: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            state["history"].append(["user", user_input])
            state["user_query"] = user_input

            for event in graph.stream(state, config):
                for value in event.values():
                    state.update(value)
                    if state.get("history"):
                        role, text = state["history"][-1]
                        if role == "ai" and text:
                            print("\nTutor:", text)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            break
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            state = initialize_state()
            print("Conversation reset.")

# ---- Comprehensive tests (like your sample) ----
def find_figures_for_topic(topic: str) -> List[Dict[str, Any]]:
    res = []
    name = find_figure_for_topic(topic)
    if name:
        path = get_figure_path(name)
        res.append({
            "figure": name,
            "desc": (FIGURE_LOOKUP.get(name) or {}).get("desc", ""),
            "path": path,
            "exists": os.path.exists(path)
        })
    return res

def run_comprehensive_test():
    print("=== COMPREHENSIVE SYSTEM TEST ===")

    test_cases = [
        {"input": "start lesson on human brain", "expected": "rag", "description": "Lesson start request"},
        {"input": "show diagram of brain", "expected": "image", "description": "Image request"},
        {"input": "doubt: what is the cerebrum?", "expected": "interrupt", "description": "Interrupt question"},
        {"input": "hello how are you?", "expected": "general", "description": "General conversation"},
    ]

    for i, t in enumerate(test_cases, 1):
        print(f"\nTest {i}: {t['description']}")
        print(f"Input: '{t['input']}'")
        state = initialize_state()
        state["messages"] = [("user", t["input"])]
        out = route_query(state)
        print(f"Expected: {t['expected']}, Got: {out['next']}")
        print(f"Status: {'✓' if out['next']==t['expected'] else '✗'}")
        if out['next'] == 'rag' and out.get('current_topic'):
            print(f"Current topic: {out['current_topic']}")

    print("\n=== IMAGE FINDING TEST ===")
    for topic in ["human brain", "nervous system", "digestion"]:
        figs = find_figures_for_topic(topic)
        print(f"Topic: '{topic}' -> Found {len(figs)} figures")
        for f in figs[:2]:
            print(f"  - {f['figure']}: {f['desc'][:50]}...")
            print(f"    Path: {f['path']}, Exists: {f['exists']}")

# ---- Main ----
if __name__ == "__main__":
    print("\nConfiguration:")
    print(" DATA_PATH           =", DATA_PATH)
    print(" CONTENT_INDEX_PATH  =", CONTENT_INDEX_PATH)
    print(" MEMORY_INDEX_PATH   =", MEMORY_INDEX_PATH)
    print(" EMBEDDING_MODEL     =", EMBEDDING_MODEL_NAME)
    print(" LLM_MODEL           =", LLM_MODEL)
    if not API_KEY:
        print(" WARNING: GROQ_API_KEY missing; set it for live LLM calls.")

    # Run tests once (comment if not needed)
    run_comprehensive_test()

    # Start REPL
    chat_with_tutor()
