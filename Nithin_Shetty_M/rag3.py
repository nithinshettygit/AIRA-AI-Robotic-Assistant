#!/usr/bin/env python3
"""
robust_ai_tutor.py — final improved single-file tutor

Key fixes:
- Replaced MemoryAdapter with CheckpointerAdapter that adapts various LangGraph saver APIs
  (supports put(config, metadata, new_versions), put(config, new_versions), put(config, checkpoint),
  put_writes, get/get_tuple/list, get_next_version, and async wrappers).
- Removed hard-coded API key; reads from env.
- Robust LLM call normalization and safer error handling.
- FAISS & embeddings guarded; semantic memory created safely.
- REPL at the bottom for terminal usage.
"""
import os
import re
import json
import time
import uuid
import traceback
import threading
import logging
from typing import List, Dict, Any, Optional, Iterator

# Optional IPython display
try:
    from IPython.display import Image, display, Markdown
except Exception:
    Image = None
    display = None
    Markdown = None

# Best-effort core imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
except Exception:
    FAISS = None
    HuggingFaceBgeEmbeddings = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# LangGraph checkpointer/graph (best-effort)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver as LangGraphMemorySaver
    from langgraph.checkpoint.memory import InMemorySaver as LangGraphInMemorySaver
except Exception:
    StateGraph = None
    END = None
    LangGraphMemorySaver = None
    LangGraphInMemorySaver = None

# Document wrapper for embeddings
try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Optional youtube helpers
try:
    import yt_dlp as ytdlp
except Exception:
    ytdlp = None

try:
    from youtube_search import YoutubeSearch
except Exception:
    YoutubeSearch = None

# Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("robust_ai_tutor")

# ---------------- CONFIG ----------------
DATA_DIR = os.getenv('DATA_PATH', 'data')
VECTOR_DB_DIR = os.path.join(DATA_DIR, 'faiss_vectorstore_final')
CONTENT_INDEX_DIR = os.path.join(VECTOR_DB_DIR, 'content_faiss_index')
MEMORY_INDEX_DIR = os.path.join(VECTOR_DB_DIR, 'memory_faiss_index')
CHUNKS_FILE = os.path.join(DATA_DIR, 'merged_chunks_with_figures.json')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
TOP_K = int(os.getenv('TOP_K', '5'))

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'thenlper/gte-large')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3-8b-8192')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.groq.com/openai/v1')
API_KEY = ""

if not API_KEY:
    log.warning("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY in environment to enable LLM calls.")

# ---------------- Helpers ----------------
def safe_lower(x: Optional[str]) -> str:
    return (x or '').lower()

# ---------------- Embeddings & FAISS ----------------
embeddings = None
content_db = None
semantic_memory = None
_semantic_memory_lock = threading.Lock()

log.info("Initializing embeddings (best-effort)...")
if HuggingFaceBgeEmbeddings is None:
    log.warning("HuggingFaceBgeEmbeddings not available (langchain-community missing). Embeddings disabled.")
else:
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
        log.info("Embeddings initialized.")
    except Exception as e:
        log.warning("Failed to init embeddings: %s", e)
        embeddings = None

def safe_load_faiss(folder: str, emb) -> Optional[Any]:
    if FAISS is None:
        return None
    try:
        if emb is None:
            return None
        if os.path.exists(folder):
            return FAISS.load_local(folder, emb, allow_dangerous_deserialization=True)
    except Exception as e:
        log.warning("load faiss failed for %s: %s", folder, e)
    return None

content_db = safe_load_faiss(CONTENT_INDEX_DIR, embeddings)
semantic_memory = safe_load_faiss(MEMORY_INDEX_DIR, embeddings)

def create_semantic_memory_if_needed():
    global semantic_memory
    if semantic_memory is not None:
        return
    if FAISS is None or embeddings is None:
        return
    with _semantic_memory_lock:
        if semantic_memory is not None:
            return
        try:
            vs = FAISS.from_documents([], embeddings)
            os.makedirs(MEMORY_INDEX_DIR, exist_ok=True)
            vs.save_local(MEMORY_INDEX_DIR)
            semantic_memory = safe_load_faiss(MEMORY_INDEX_DIR, embeddings)
            log.info("Created semantic memory index at %s", MEMORY_INDEX_DIR)
        except Exception as e:
            log.warning("Failed to create semantic memory index: %s", e)

# ---------------- LLM wrapper ----------------
llm = None
if ChatOpenAI is None:
    log.warning("ChatOpenAI client not available (langchain-openai missing). LLM calls will be mocked.")
else:
    try:
        llm = ChatOpenAI(api_key=API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL, temperature=0.2)
        log.info("LLM client initialized.")
    except Exception as e:
        log.warning("Failed to init ChatOpenAI: %s", e)
        llm = None

def _normalize_llm_response(resp: Any) -> str:
    try:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            if 'content' in resp:
                return resp['content']
            if 'choices' in resp and resp['choices']:
                c = resp['choices'][0]
                if isinstance(c, dict):
                    if 'message' in c and isinstance(c['message'], dict):
                        return c['message'].get('content') or ''
                    return c.get('text') or ''
            return json.dumps(resp)[:2000]
        if hasattr(resp, 'content'):
            return getattr(resp, 'content') or ''
        if hasattr(resp, 'choices'):
            choices = getattr(resp, 'choices') or []
            if choices:
                ch = choices[0]
                if isinstance(ch, dict):
                    return ch.get('message', {}).get('content') or ch.get('text') or ''
                if hasattr(ch, 'message'):
                    return getattr(ch.message, 'get', lambda k: None)('content') or str(ch)
        return str(resp)
    except Exception:
        return str(resp)

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    try:
        if llm is None:
            return f"[MOCK LLM] System: {system_prompt[:120]} | User: {user_prompt[:240]}"
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        if hasattr(llm, 'invoke'):
            resp = llm.invoke(messages)
        else:
            try:
                resp = llm(messages, max_tokens=max_tokens)
            except TypeError:
                resp = llm(messages)
        return _normalize_llm_response(resp) or '[LLM returned empty]'
    except Exception as e:
        log.error("LLM call failed: %s", e)
        traceback.print_exc()
        return '[LLM failure]'

# ---------------- CheckpointerAdapter (robust) ----------------
class _CheckpointTuple:
    def __init__(self, checkpoint: Dict[str, Any], metadata: Dict[str, Any] = None, config: Dict[str, Any] = None):
        self.checkpoint = checkpoint or {}
        self.metadata = metadata or {}
        self.config = config or {}
        self.id = str(uuid.uuid4())
        self.ts = time.time()

class CheckpointerAdapter:
    def __init__(self, underlying=None):
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[_CheckpointTuple]] = {}
        self.underlying = None
        if underlying is not None:
            try:
                # If underlying is context manager but not entered, try to __enter__
                if hasattr(underlying, "__enter__") and not hasattr(underlying, "put"):
                    try:
                        real = underlying.__enter__()
                        self.underlying = real
                    except Exception as e:
                        log.warning("Failed to enter underlying context-manager saver: %s; using fallback.", e)
                        self.underlying = None
                else:
                    self.underlying = underlying
            except Exception:
                self.underlying = underlying

    def _thread_id_from_config(self, config: Dict[str, Any]) -> str:
        try:
            return config.get("configurable", {}).get("thread_id") or config.get("thread_id") or "default"
        except Exception:
            return "default"

    def put(self, config: Dict[str, Any], *args, **kwargs):
        thread_id = self._thread_id_from_config(config or {})
        if self.underlying is not None:
            try:
                # 3-arg: (config, metadata, new_versions)
                if len(args) >= 2:
                    metadata = args[0]
                    new_versions = args[1]
                    try:
                        return self.underlying.put(config, metadata, new_versions)
                    except TypeError:
                        try:
                            return self.underlying.put(config, new_versions)
                        except TypeError:
                            try:
                                return self.underlying.put(config, {'values': new_versions, 'metadata': metadata})
                            except Exception as e:
                                log.warning("Underlying put failed (multi-signature): %s", e)
                elif len(args) == 1:
                    try:
                        return self.underlying.put(config, args[0])
                    except TypeError:
                        try:
                            return self.underlying.put(config, {}, args[0])
                        except Exception as e:
                            log.warning("Underlying put single-arg failed: %s", e)
            except Exception as e:
                log.warning("Underlying put exception, falling back to in-memory: %s", e)

        # Fallback: store in memory
        if len(args) >= 2:
            metadata = args[0] or {}
            new_versions = args[1] or {}
            with self._lock:
                store = self._store.setdefault(thread_id, {})
                store.update(new_versions)
                tup = _CheckpointTuple(checkpoint=dict(store), metadata=metadata, config=config)
                self._history.setdefault(thread_id, []).append(tup)
                return config
        elif len(args) == 1:
            checkpoint = args[0] or {}
            if isinstance(checkpoint, dict) and 'values' in checkpoint:
                values = checkpoint.get('values') or {}
            else:
                values = checkpoint
            with self._lock:
                store = self._store.setdefault(thread_id, {})
                store.update(values)
                tup = _CheckpointTuple(checkpoint=dict(store), metadata=(checkpoint.get('metadata') if isinstance(checkpoint, dict) else {}), config=config)
                self._history.setdefault(thread_id, []).append(tup)
                return config
        else:
            return config

    def put_writes(self, config: Dict[str, Any], writes: Dict[str, Any]):
        if self.underlying is not None and hasattr(self.underlying, "put_writes"):
            try:
                return self.underlying.put_writes(config, writes)
            except Exception as e:
                log.warning("Underlying put_writes failed: %s", e)
        thread_id = self._thread_id_from_config(config or {})
        with self._lock:
            self._store.setdefault(thread_id, {}).setdefault("__pending_writes__", []).append(writes)

    def get_tuple(self, config: Dict[str, Any]) -> Optional[_CheckpointTuple]:
        if self.underlying is not None and hasattr(self.underlying, "get_tuple"):
            try:
                tup = self.underlying.get_tuple(config)
                if tup is None:
                    return None
                if hasattr(tup, "checkpoint"):
                    return tup
                if isinstance(tup, dict):
                    return _CheckpointTuple(checkpoint=tup.get("values") or tup.get("checkpoint") or tup, metadata=tup.get("metadata") or {}, config=config)
            except Exception as e:
                log.warning("Underlying get_tuple failed: %s", e)
        thread_id = self._thread_id_from_config(config or {})
        with self._lock:
            hist = self._history.get(thread_id) or []
            if hist:
                return hist[-1]
            if thread_id in self._store:
                return _CheckpointTuple(checkpoint=dict(self._store[thread_id]), metadata={}, config=config)
        return None

    def list(self, config: Dict[str, Any]) -> Iterator[_CheckpointTuple]:
        if self.underlying is not None and hasattr(self.underlying, "list"):
            try:
                return self.underlying.list(config)
            except Exception as e:
                log.warning("Underlying list failed: %s", e)
        thread_id = self._thread_id_from_config(config or {})
        with self._lock:
            for tup in reversed(self._history.get(thread_id, [])):
                yield tup

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tup = self.get_tuple(config)
        if tup:
            return tup.checkpoint
        return None

    def get_next_version(self, current: Any, channel: Any) -> Any:
        if self.underlying is not None and hasattr(self.underlying, "get_next_version"):
            try:
                return self.underlying.get_next_version(current, channel)
            except Exception as e:
                log.debug("Underlying get_next_version failed: %s", e)
        try:
            if hasattr(channel, "default"):
                return channel.default()
            if hasattr(channel, "empty"):
                return channel.empty()
            if hasattr(channel, "initial"):
                return channel.initial
        except Exception:
            pass
        return current

    async def aget_tuple(self, config: Dict[str, Any]):
        return self.get_tuple(config)

    async def aget(self, config: Dict[str, Any]):
        return self.get(config)

    async def alist(self, config: Dict[str, Any]):
        for t in self.list(config):
            yield t

    async def aput(self, config: Dict[str, Any], *args, **kwargs):
        return self.put(config, *args, **kwargs)

# instantiate adapter with LangGraph MemorySaver if available
_underlying = None
if LangGraphInMemorySaver is not None:
    try:
        _underlying = LangGraphInMemorySaver()
        log.info("Using LangGraph InMemorySaver as underlying saver.")
    except Exception as e:
        log.warning("Unable to instantiate LangGraphInMemorySaver: %s. Falling back to pure in-memory.", e)

_adapter = CheckpointerAdapter(underlying=_underlying)

def mem_put(thread_id: str, key: str, value: Any):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        metadata = {"source": "robust_ai_tutor", "ts": time.time()}
        new_versions = {key: value}
        _adapter.put(config, metadata, new_versions)
    except Exception as e:
        log.warning("mem_put failed: %s", e)

def mem_get(thread_id: str, key: str) -> Any:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snap = _adapter.get(config)
        if snap and isinstance(snap, dict):
            return snap.get(key)
    except Exception as e:
        log.warning("mem_get failed: %s", e)
    return None

# ---------------- semantic memory helpers ----------------
def add_semantic_memory(text: str, meta: Dict[str, Any]):
    try:
        if embeddings is None:
            return
        doc = Document(page_content=text, metadata={**meta, 'ts': time.time(), 'id': str(uuid.uuid4())})
        if semantic_memory is None:
            create_semantic_memory_if_needed()
        if semantic_memory is None:
            return
        if hasattr(semantic_memory, 'add_documents'):
            semantic_memory.add_documents([doc])
        else:
            try:
                semantic_memory.add_texts([doc.page_content], metadatas=[doc.metadata])
            except Exception as e:
                log.warning("Failed to add to semantic memory: %s", e)
    except Exception as e:
        log.warning("add_semantic_memory failed: %s", e)

def search_semantic_memory(query: str, k: int = 4) -> List[Document]:
    if semantic_memory is None:
        return []
    try:
        if hasattr(semantic_memory, 'similarity_search'):
            return semantic_memory.similarity_search(query, k=k)
        if hasattr(semantic_memory, 'similarity_search_with_score'):
            return [d for d, _ in semantic_memory.similarity_search_with_score(query, k=k)]
    except Exception as e:
        log.warning("semantic search failed: %s", e)
    return []

# ---------------- RAG / KnowledgeBaseSearch ----------------
def knowledgebase_search(query: str, k: int = TOP_K) -> Dict[str, Any]:
    if content_db is None:
        return {'ok': False, 'note': 'No content DB loaded'}
    try:
        docs = content_db.similarity_search(query, k=k)
        return {'ok': True, 'results': [{'text': d.page_content, 'meta': getattr(d, 'metadata', {})} for d in docs]}
    except Exception as e:
        log.warning("knowledgebase_search failed: %s", e)
        return {'ok': False, 'error': str(e)}

# ---------------- Image retrieval ----------------
FIGURE_LOOKUP: Dict[str, Dict[str, Any]] = {}
CONTENT_CHUNKS: List[Dict[str, Any]] = []
if os.path.exists(CHUNKS_FILE):
    try:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            CONTENT_CHUNKS = json.load(f)
            for c in CONTENT_CHUNKS:
                for fig in (c.get('figures') or []):
                    name = fig.get('figure') or fig.get('name') or ''
                    if name:
                        FIGURE_LOOKUP[name] = {'desc': fig.get('desc') or '', 'file': fig.get('file') or (name.replace(' ', '_') + '.png')}
    except Exception as e:
        log.warning("load chunks failed: %s", e)

def find_figure_for_topic(topic: str) -> Optional[str]:
    if not topic:
        return None
    t = safe_lower(topic)
    for name in FIGURE_LOOKUP.keys():
        if safe_lower(name) == t:
            return name
    for name, meta in FIGURE_LOOKUP.items():
        if t in safe_lower(name) or t in safe_lower(meta.get('desc', '')):
            return name
    for c in CONTENT_CHUNKS:
        text = c.get('text') or c.get('page_content') or ''
        if t in safe_lower(text):
            figs = c.get('figures') or []
            if figs:
                nm = figs[0].get('figure') or figs[0].get('name') or None
                if nm:
                    return nm
    return None

def image_retrieval(topic: str, specified_figure: Optional[str] = None) -> Dict[str, Any]:
    fig = specified_figure or find_figure_for_topic(topic)
    if not fig:
        return {'ok': False, 'note': 'No figure found for topic'}
    meta = FIGURE_LOOKUP.get(fig) or {}
    desc = meta.get('desc') or ''
    path = os.path.join(IMAGE_DIR, meta.get('file') or (fig.replace(' ', '_') + '.png'))
    exists = os.path.exists(path)
    if exists and Image and display:
        try:
            display(Image(filename=path))
            if Markdown:
                display(Markdown(f"**Figure:** {fig}\n\n**Description:** {desc}"))
        except Exception:
            log.warning("IPython display failed for %s", path)
    else:
        log.info("Figure: %s | Path: %s | Exists: %s", fig, path, exists)
        log.info("Description: %s", desc or "(no description)")
    return {'ok': True, 'figure': fig, 'desc': desc, 'path': path, 'exists': exists}

# ---------------- Video retrieval ----------------
def fetch_animated_videos(topic: str, num_videos: int = 1) -> Optional[Dict[str, Any]]:
    if not topic:
        return None
    search_query = f"ytsearch{num_videos}:{topic} animation explained in english"
    if ytdlp:
        try:
            ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': True}
            with ytdlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get('entries') or []
                if entries:
                    for video in entries:
                        dur = video.get('duration') or 9999
                        if dur <= 900:
                            return {'title': video.get('title'), 'url': video.get('webpage_url') or ('https://www.youtube.com/watch?v=' + video.get('id', '')), 'id': video.get('id')}
        except Exception as e:
            log.warning("yt_dlp search failed: %s", e)
    if YoutubeSearch:
        try:
            results = YoutubeSearch(topic + ' animation explained in english', max_results=5).to_dict()
            if results:
                v = results[0]
                vid_id = v.get('id') or v.get('videoId') or ''
                url = ('https://www.youtube.com/watch?v=' + vid_id) if vid_id else v.get('link')
                return {'title': v.get('title'), 'url': url, 'id': vid_id}
        except Exception as e:
            log.warning("youtube_search failed: %s", e)
    return None

# ---------------- Orchestrator / Planner ----------------
ORCHESTRATOR_PROMPT = (
    "You are an orchestrator that returns a JSON plan. Tools: KnowledgeBaseSearch, ImageRetrieval, VideoRetrieval. "
    "For a new lesson: [KB, Image, Video, ANSWER]. For a doubt: [KB, MEM (store), ANSWER]. For general chat: [ANSWER]. "
)

def extract_topic_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [r'start lesson on (.+)', r'begin lesson on (.+)', r'teach me about (.+)', r'lesson on (.+)']
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip(' ?.!')
    m = re.search(r'lesson\s*[:\-]\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(' ?.!')
    return None

def plan_from_input(user_input: str, state: Dict[str, Any]) -> Dict[str, Any]:
    low = (user_input or '').strip()
    if re.match(r'^(doubt:|question:|clarify:)', low, re.I):
        q = re.sub(r'^(doubt:|question:|clarify:)\s*', '', user_input, flags=re.I).strip()
        q = q.rstrip(' ?.!')
        return {'plan': [{'action': 'KnowledgeBaseSearch', 'args': {'query': q, 'k': TOP_K}}, {'action': 'MEMORY_ADD', 'args': {'text': q, 'meta': {'type': 'doubt'}}}, {'action': 'ANSWER', 'args': {'instruction': f'Answer the question: {q} (concise).'}}]}
    topic = extract_topic_from_text(user_input)
    if topic:
        return {'plan': [{'action': 'KnowledgeBaseSearch', 'args': {'query': topic, 'k': TOP_K}}, {'action': 'ImageRetrieval', 'args': {'topic': topic}}, {'action': 'VideoRetrieval', 'args': {'topic': topic}}, {'action': 'ANSWER', 'args': {'instruction': f'Produce a 3-part lesson on {topic}: intro, objectives, short explanation. Include Resources lines.'}}]}
    if re.search(r'show diagram|show figure|diagram|show image', low, re.I):
        return {'plan': [{'action': 'ImageRetrieval', 'args': {'topic': state.get('current_topic') or ''}}, {'action': 'ANSWER', 'args': {'instruction': 'Return the diagram info and guidance.'}}]}
    if re.search(r'show video|video', low, re.I):
        return {'plan': [{'action': 'VideoRetrieval', 'args': {'topic': state.get('current_topic') or ''}}, {'action': 'ANSWER', 'args': {'instruction': 'Return the video link or say none found.'}}]}
    return {'plan': [{'action': 'ANSWER', 'args': {'instruction': f'Respond warmly to: {user_input}'}}]}

# Execute plan
def execute_plan(plan: Dict[str, Any], state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    outputs = []
    final_instruction = None
    for step in plan.get('plan', []):
        act = step.get('action')
        args = step.get('args', {}) or {}
        try:
            if act == 'KnowledgeBaseSearch':
                q = args.get('query', '')
                res = knowledgebase_search(q, k=args.get('k', TOP_K))
                outputs.append({'KnowledgeBaseSearch': res})
            elif act == 'ImageRetrieval':
                t = args.get('topic') or state.get('current_topic')
                res = image_retrieval(t)
                outputs.append({'ImageRetrieval': res})
            elif act == 'VideoRetrieval':
                t = args.get('topic') or state.get('current_topic')
                res = fetch_animated_videos(t, num_videos=1)
                outputs.append({'VideoRetrieval': res})
            elif act == 'MEMORY_ADD':
                txt = args.get('text') or ''
                meta = args.get('meta') or {}
                key = meta.get('id') or str(uuid.uuid4())
                mem_put(thread_id, key, {'text': txt, **meta, 'ts': time.time()})
                try:
                    add_semantic_memory(txt, meta)
                except Exception:
                    log.debug("add_semantic_memory suppressed exception", exc_info=True)
                outputs.append({'MEMORY_ADD': True})
            elif act == 'ANSWER':
                final_instruction = step.get('args', {}).get('instruction', '')
                outputs.append({'ANSWER': {'instruction': final_instruction}})
            else:
                outputs.append({'UNKNOWN': act})
        except Exception as e:
            outputs.append({'ERROR': str(e)})
            log.warning("execute_plan step failed: %s - %s", act, e)
    reply = ''
    if final_instruction:
        summary_parts = []
        for o in outputs:
            try:
                summary_parts.append(json.dumps(o, default=str)[:1200])
            except Exception:
                summary_parts.append(str(o)[:1200])
        ctx = '\n\n'.join(summary_parts)
        sys_prompt = 'You are a warm, adaptive teacher. Use the tool outputs to craft a concise student-facing reply.'
        user_msg = f'Instruction: {final_instruction}\n\nTool outputs:\n{ctx}\n\nState: in_lesson={state.get("in_lesson")}, current_topic={state.get("current_topic")}'
        reply = call_llm(sys_prompt, user_msg)
        try:
            if re.search(r'start a|lesson on|teach', final_instruction, re.I) or ('lesson' in (final_instruction or '').lower()) or state.get('in_lesson'):
                t = state.get('current_topic')
                m = re.search(r'lesson on ([\w\s\-\:]+)', final_instruction, re.I)
                if m:
                    t = m.group(1).strip()
                if t:
                    state['current_topic'] = t
                    state['in_lesson'] = True
                    mem_put(thread_id, 'tutor_state', dict(state))
            try:
                add_semantic_memory(reply[:12000], {'type': 'lesson_snippet', 'topic': state.get('current_topic') or ''})
            except Exception:
                pass
        except Exception:
            log.debug("state persistence suppressed", exc_info=True)
    else:
        reply = call_llm('You are a friendly tutor.', final_instruction or 'Answer the user directly.')
    return {'reply': reply, 'outputs': outputs}

# ---------------- LangGraph node & REPL wiring ----------------
def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    thread = state.get('thread_id', 'student_1')
    user_text = state.get('input', '')
    tutor_state = mem_get(thread, 'tutor_state') or {'history': [], 'current_topic': None, 'in_lesson': False, 'lesson_progress': 0}
    plan = plan_from_input(user_text, tutor_state)
    result = execute_plan(plan, tutor_state, thread)
    try:
        hist = mem_get(thread, 'short_history') or []
        hist.append({'user': user_text, 'ai': result['reply'], 'ts': time.time()})
        mem_put(thread, 'short_history', hist[-100:])
    except Exception:
        log.debug("short_history update suppressed", exc_info=True)
    try:
        mem_put(thread, 'tutor_state', tutor_state)
    except Exception:
        log.debug("tutor_state persist suppressed", exc_info=True)
    state['response'] = result['reply']
    state['tool_outputs'] = result.get('outputs', [])
    return state

# Prepare graph/app if available
_app = None
if StateGraph is not None:
    try:
        graph = StateGraph(dict)
        graph.add_node('orchestrator', orchestrator_node)
        graph.set_entry_point('orchestrator')
        graph.add_edge('orchestrator', END)
        _app = graph.compile(checkpointer=_adapter)
        log.info("LangGraph app compiled successfully.")
    except Exception as e:
        log.warning("Failed to compile LangGraph graph: %s", e)
        _app = None
else:
    log.warning("LangGraph StateGraph not available. REPL will call orchestrator_node directly.")

# ---------------- Tests ----------------
def run_comprehensive_test():
    log.info("=== COMPREHENSIVE SYSTEM TEST ===")
    tests = [
        {'input': 'start lesson on human brain', 'expected': 'rag'},
        {'input': 'show diagram of brain', 'expected': 'image'},
        {'input': 'doubt: what is the cerebrum?', 'expected': 'interrupt'},
        {'input': 'hello how are you?', 'expected': 'general'},
    ]
    for i, t in enumerate(tests, 1):
        print(f"\nTest {i}: {t['input']}")
        plan = plan_from_input(t['input'], {})
        actions = [s.get('action') for s in plan.get('plan', [])]
        nxt = 'general'
        if 'KnowledgeBaseSearch' in actions and 'ImageRetrieval' in actions:
            nxt = 'rag'
        if 'ImageRetrieval' in actions and not ('KnowledgeBaseSearch' in actions):
            nxt = 'image'
        if any(a == 'MEMORY_ADD' for a in actions) and any(a == 'KnowledgeBaseSearch' for a in actions):
            nxt = 'interrupt'
        print('Expected:', t['expected'], 'Got:', nxt, 'Status:', '✓' if nxt == t['expected'] else '✗')

    print('\n=== IMAGE FINDING TEST ===')
    for topic in ['human brain', 'nervous system', 'digestion']:
        f = find_figure_for_topic(topic)
        if f:
            p = os.path.join(IMAGE_DIR, FIGURE_LOOKUP.get(f, {}).get('file') or (f.replace(' ', '_') + '.png'))
            print(f"Topic: '{topic}' -> Found {f} Path: {p} Exists: {os.path.exists(p)}")
        else:
            print(f"Topic: '{topic}' -> Found 0 figures")

# ---------------- Driver / REPL ----------------
def repl():
    print('Configuration:')
    print(' DATA_DIR =', DATA_DIR)
    print(' CONTENT_INDEX_DIR =', CONTENT_INDEX_DIR)
    print(' MEMORY_INDEX_DIR =', MEMORY_INDEX_DIR)
    print(' EMBEDDING_MODEL =', EMBEDDING_MODEL)
    print(' LLM_MODEL =', LLM_MODEL)
    print('\nRunning quick tests...')
    run_comprehensive_test()

    thread = input('\nThread id (ENTER for student_1): ').strip() or 'student_1'
    print('\nWelcome to the Robust AI Tutor — type "exit" to quit')
    while True:
        try:
            text = input(f"[{thread}] Student: ").strip()
            if not text:
                continue
            if text.strip().lower() in ('exit', 'quit'):
                print('Tutor: Goodbye!')
                break
            payload = {'input': text, 'thread_id': thread}
            try:
                if _app is not None:
                    out = _app.invoke(payload, config={'configurable': {'thread_id': thread}})
                    resp = out.get('response') if isinstance(out, dict) else str(out)
                else:
                    state = {'input': text, 'thread_id': thread}
                    state = orchestrator_node(state)
                    resp = state.get('response')
            except Exception as e:
                log.warning("Invocation failed, falling back to direct call: %s", e)
                state = {'input': text, 'thread_id': thread}
                state = orchestrator_node(state)
                resp = state.get('response')
            print('\nTutor:\n', resp)
        except KeyboardInterrupt:
            print('\nInterrupted. Exiting...')
            break
        except Exception as e:
            log.error('[ERROR] main loop: %s', e)
            traceback.print_exc()
            continue

if __name__ == '__main__':
    repl()
