"""
MAVI AI Agent - Enhanced & Unified Version (fixed)
Changes applied:
- Robust SQLite (WAL, check_same_thread=False, safe per-call connections)
- Proper Gemini configuration: set GOOGLE_API_KEY env var, use genai.configure(api_key=...)
- Prioritize gemini-2.5-pro model
- Improved error handling for DB and Gemini

Run:
1. pip install -r requirements.txt
   (requirements.txt should include: streamlit openai google-generativeai pytesseract PyPDF2 python-dotenv requests numpy pillow)
2. Put keys in .env or Streamlit secrets (GEMINI_API_KEY, NEWS_API_KEY, QWEN_BASE_URL/QWEN_API_KEY if using local Qwen)
3. streamlit run stream.py
"""

import os
import io
import base64
import json
import time
import sqlite3
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import requests
import streamlit as st
from PIL import Image
import pytesseract
import PyPDF2

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Import LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ============================================================================
# CONFIGURATION & SECRETS
# ============================================================================

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read secrets from environment, Streamlit secrets, or return default."""
    val = os.environ.get(name)
    if val:
        return val.strip()
    try:
        if name in st.secrets:
            val = st.secrets[name]
            return val.strip() if isinstance(val, str) else val
    except Exception:
        pass
    return default

# API Keys and endpoints
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")
QWEN_BASE_URL = get_secret("QWEN_BASE_URL", "http://localhost:12434/engines/v1")
QWEN_API_KEY = get_secret("QWEN_API_KEY", "docker")
LLM_MODEL = get_secret("LLM_MODEL", "ai/qwen2.5:3B-Q4_K_M")
EMBED_MODEL = get_secret("EMBED_MODEL", "ai/snowflake-arctic-embed-l-v2")

# Memory files
MEMORY_JSON = "agent_memory.json"
MEMORY_DB = "agent_memory.db"

# Initialize clients
client = None
if OPENAI_AVAILABLE:
    try:
        client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
    except Exception as e:
        # avoid calling streamlit APIs too early in some environments
        try:
            st.warning(f"Local LLM client initialization failed: {e}")
        except Exception:
            print("Local LLM client initialization failed:", e)

# Configure Gemini safely
GEMINI_READY = False
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        # ensure GOOGLE_API_KEY env var exists for libraries that check it
        os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
        # configure genai using the correct keyword
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except TypeError:
            # older/newer clients might require configure(GOOGLE_API_KEY=...)
            try:
                genai.configure(GOOGLE_API_KEY=GEMINI_API_KEY)
            except Exception:
                pass
        GEMINI_READY = True
    except Exception as e:
        GEMINI_READY = False
        try:
            st.warning(f"Gemini configure failed: {e}")
        except Exception:
            print("Gemini configure failed:", e)

# ============================================================================
# MEMORY MANAGER (robust for Streamlit)
# ============================================================================

class MemoryManager:
    """Persistent memory with JSON for conversations and SQLite for key-value storage."""
    def __init__(self, json_path: str = MEMORY_JSON, db_path: str = MEMORY_DB):
        self.json_path = json_path
        self.db_path = db_path
        self._init_json()
        self._init_db()

    def _init_json(self):
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump({"conversations": []}, f)

    def _init_db(self):
        # Use WAL and allow cross-thread connections for Streamlit
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        try:
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")
            conn.commit()
        except Exception:
            pass
        conn.close()

    def append_conversation(self, role: str, text: str):
        try:
            with open(self.json_path, "r+") as f:
                data = json.load(f)
                data.setdefault("conversations", []).append({
                    "role": role,
                    "text": text,
                    "ts": time.time()
                })
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception:
            with open(self.json_path, "w") as f:
                json.dump({"conversations": [{
                    "role": role,
                    "text": text,
                    "ts": time.time()
                }]}, f, indent=2)

    def get_recent_conversation(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
                return data.get("conversations", [])[-limit:]
        except Exception:
            return []

    def store_kv(self, key: str, value: Any):
        try:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            c = conn.cursor()
            c.execute("INSERT INTO memories (key, value) VALUES (?, ?)", (key, json.dumps(value)))
            conn.commit()
            conn.close()
        except Exception as e:
            # fallback: log and append to json
            try:
                print("SQLite store_kv error:", e)
            except Exception:
                pass
            try:
                with open(self.json_path, "r+") as f:
                    data = json.load(f)
                    data.setdefault("conversations", []).append({"role": "system", "text": f"SQLite error: {e}", "ts": time.time()})
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
            except Exception:
                pass

    def query_kv(self, key: str) -> List[Tuple[int, str]]:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT id, value FROM memories WHERE key = ? ORDER BY id DESC", (key,))
            rows = c.fetchall()
            conn.close()
            return rows
        except Exception as e:
            try:
                print("SQLite query_kv error:", e)
            except Exception:
                pass
            return []

# ============================================================================
# COMMAND PARSER
# ============================================================================

class CommandParser:
    """Parse slash commands like /calc, /news, /echo."""
    def parse(self, text: str) -> Tuple[Optional[str], str]:
        text = text.strip()
        if not text:
            return None, ""
        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd = parts[0][1:]
            arg = parts[1] if len(parts) > 1 else ""
            return cmd.lower(), arg
        return None, text

# ============================================================================
# DOCUMENT PROCESSING HELPERS
# ============================================================================

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text()
                if t:
                    pages.append(t)
            except Exception:
                pass
        return "\n\n".join(pages)
    except Exception:
        return ""

def extract_text_from_image_bytes(b: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

# ============================================================================
# EMBEDDING & RAG FUNCTIONS
# ============================================================================

def call_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings from local model."""
    if client is None:
        return []
    try:
        res = client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeds = []
        for item in res.data:
            emb = None
            try:
                emb = getattr(item, "embedding", None)
            except Exception:
                pass
            if emb is None:
                try:
                    emb = item.get("embedding") if isinstance(item, dict) else None
                except Exception:
                    pass
            if emb is None:
                try:
                    emb = item["embedding"]
                except Exception:
                    pass
            if emb is not None:
                embeds.append(np.array(emb, dtype=np.float32))
        return embeds
    except Exception as e:
        try:
            st.error(f"Embedding error: {e}")
        except Exception:
            print("Embedding error:", e)
        return []

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def retrieve_top_k(query: str, texts: List[str], embeds: List[np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
    if not texts or not embeds:
        return []
    q_embs = call_embeddings([query])
    if not q_embs:
        return []
    q = q_embs[0]
    scores = [cosine_sim(q, e) for e in embeds]
    idx = np.argsort(scores)[::-1][:k]
    return [(texts[i], float(scores[i])) for i in idx]

# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def stream_local_llm(messages: List[dict]):
    """Stream response from local LLM."""
    if client is None:
        yield "[ERROR] Local LLM client not configured."
        return
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            try:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
            except Exception:
                pass
    except Exception as e:
        yield f"[ERROR] {e}"

# ============================================================================
# GEMINI FUNCTIONS (prioritize gemini-2.5-pro)
# ============================================================================

def analyze_image_with_gemini(image_bytes: bytes, prompt: str = "Describe this image in detail") -> str:
    if not GEMINI_READY:
        return "Gemini not configured. Set GEMINI_API_KEY in secrets or environment."
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model_names = [
            "gemini-2.5-pro",
            "gemini-2.5",
            "gemini-2.5-vision",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro",
        ]
        last_error = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # some clients accept mixed content; pass prompt and image
                response = model.generate_content([prompt, img])
                return response.text
            except Exception as e:
                last_error = e
                continue
        return f"Gemini image analysis error: {last_error}"
    except Exception as e:
        return f"Gemini image analysis error: {e}"

def call_gemini_text(prompt: str, max_output_tokens: int = 512) -> str:
    if not GEMINI_READY:
        return "Gemini not configured."
    try:
        model_names = [
            "gemini-2.5-pro",
            "gemini-2.5",
            "gemini-2.5-vision",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-pro",
        ]
        last_error = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_output_tokens,
                    )
                )
                return response.text
            except Exception as e:
                last_error = e
                continue
        return f"Gemini error: All models failed. Last error: {last_error}"
    except Exception as e:
        return f"Gemini error: {e}"

# ============================================================================
# NEWS FUNCTIONS
# ============================================================================

def fetch_news_via_api(query: str, page_size: int = 5):
    if not NEWS_API_KEY:
        return {"error": "NEWS_API_KEY not configured."}
    api_key = NEWS_API_KEY.strip()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        articles = [{
            "title": a.get("title"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "url": a.get("url"),
            "description": a.get("description"),
            "content": a.get("content")
        } for a in data.get("articles", [])]
        return {"articles": articles}
    except Exception as e:
        return {"error": f"News API error: {e}"}

# ============================================================================
# TOOL SYSTEM
# ============================================================================

class ToolRouter:
    def __init__(self):
        self.tools: Dict[str, Callable[[str], str]] = {}
        self.descriptions: Dict[str, str] = {}
    def register(self, name: str, fn: Callable[[str], str], description: str = ""):
        self.tools[name] = fn
        self.descriptions[name] = description
    def run(self, name: str, arg: str) -> str:
        fn = self.tools.get(name)
        if not fn:
            return f"Tool '{name}' not found. Available: {', '.join(self.tools.keys())}"
        try:
            return fn(arg)
        except Exception as e:
            return f"Tool error in '{name}': {e}"
    def list_tools(self) -> List[Tuple[str, str]]:
        return [(name, self.descriptions.get(name, "")) for name in self.tools]

# tools
def tool_calc(arg: str) -> str:
    allowed_chars = "0123456789+-*/(). %"
    if any(c not in allowed_chars for c in arg):
        return "Calculator: only digits and +-*/().% allowed"
    try:
        expr = arg.replace("%", "/100")
        result = eval(expr, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

def tool_echo(arg: str) -> str:
    return f"Echo: {arg}"

def tool_news(arg: str) -> str:
    query = arg if arg else "technology"
    result = fetch_news_via_api(query, page_size=5)
    if result.get("error"):
        return result["error"]
    articles = result.get("articles", [])
    if not articles:
        return "No news articles found."
    lines = ["📰 Top Headlines:\n"]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a.get('title')} ({a.get('source')})")
        lines.append(f"   {a.get('url')}\n")
    return "\n".join(lines)

# ============================================================================
# AGENT CLASS
# ============================================================================

class MAVIAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.parser = CommandParser()
        self.tools = ToolRouter()
        self.tools.register("calc", tool_calc, "Calculator: /calc 2+2*3/(1-5)+ 4")
        self.tools.register("echo", tool_echo, "Echo: /echo message")
        self.tools.register("news", tool_news, "News: /news [query]")
        self.corpus_texts: List[str] = []
        self.corpus_embeds: List[np.ndarray] = []

    def handle_input(self, user_text: str, use_rag: bool = True, use_gemini_fallback: bool = True) -> str:
        cmd, body = self.parser.parse(user_text)
        if cmd:
            response = self.tools.run(cmd, body)
            self.memory.append_conversation("user", user_text)
            self.memory.append_conversation("tool", response)
            return response
        context = ""
        if use_rag and self.corpus_texts:
            retrieved = retrieve_top_k(user_text, self.corpus_texts, self.corpus_embeds, k=3)
            if retrieved:
                context_parts = [f"[Doc {i+1} | relevance={s:.2f}]:\n{t[:600]}" 
                               for i, (t, s) in enumerate(retrieved)]
                context = "\n\n".join(context_parts)
        recent = self.memory.get_recent_conversation(limit=6)
        memory_lines = [f"{m['role']}: {m['text'][:200]}" for m in recent]
        memory_context = "\n".join(memory_lines)
        system_prompt = "You are MAVI, a helpful and knowledgeable AI assistant. Be concise and accurate."
        user_prompt = user_text
        if context:
            user_prompt = f"Relevant documents:\n{context}\n\nUser question: {user_text}"
        if memory_context:
            user_prompt = f"Previous context:\n{memory_context}\n\n{user_prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        self.memory.append_conversation("user", user_text)
        response = ""
        for chunk in stream_local_llm(messages):
            response += chunk
        if (not response.strip() or "[ERROR]" in response) and use_gemini_fallback and GEMINI_READY:
            response = call_gemini_text(user_prompt, max_output_tokens=800)
        self.memory.append_conversation("assistant", response)
        return response

    def add_documents(self, texts: List[str]):
        if not texts:
            return
        embeds = call_embeddings(texts)
        if embeds:
            self.corpus_texts.extend(texts)
            self.corpus_embeds.extend(embeds)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MAVI AI Agent", page_icon="🤖", layout="wide")

# Styling
st.markdown("""
<style>
body { font-family: 'Inter', sans-serif; }
.title { text-align: center; font-size: 42px; font-weight: 800; 
         background: linear-gradient(90deg, #667eea, #764ba2);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         margin-bottom: 5px; }
.subtitle { text-align: center; color: #555; font-size: 14px; margin-bottom: 20px; }
.warning-box { background: #fff3cd; border: 1px solid #ffc107; padding: 12px;
               border-radius: 8px; text-align: center; color: #856404; 
               font-weight: 600; margin-bottom: 20px; }
.msg-user { background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            padding: 12px; border-radius: 10px; margin-bottom: 8px; }
.msg-assistant { background: #f5f5f5; color: #222; padding: 12px;
                 border-radius: 10px; margin-bottom: 8px; }
.msg-text { font-size: 14px; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤖 MAVI AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Local LLM • Gemini Vision • Document RAG • News • Memory</div>", unsafe_allow_html=True)
st.markdown("<div class='warning-box'>⚠️ AI responses may contain errors. Always verify important information.</div>", unsafe_allow_html=True)

# Initialize agent
if "agent" not in st.session_state:
    st.session_state.agent = MAVIAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.agent

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_gemini = st.checkbox("Enable Gemini fallback", value=GEMINI_READY)
    use_rag = st.checkbox("Enable document RAG", value=True)
    use_news = st.checkbox("Enable news fetching", value=bool(NEWS_API_KEY))
    
    st.markdown("---")
    st.markdown("### 🔑 API Status")
    st.write("Local LLM:", "☑️" if client else "❌")
    st.write("Gemini:", "☑️" if GEMINI_READY else "❌")
    st.write("News API:", "☑️" if NEWS_API_KEY else "❌")
    
    st.markdown("---")
    st.markdown("### 🛠️ Available Tools")
    for name, desc in agent.tools.list_tools():
        st.write(f"**/{name}**: {desc}")
    
    st.markdown("---")
    if st.button("🧪 Test APIs"):
        test_results = {}
        
        # Test Gemini with model discovery
        if GEMINI_READY:
            try:
                model_names = [
                    "gemini-2.5-pro"
                ]
                working_model = None
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content("Say 'Hello!'")
                        working_model = model_name
                        test_results["Gemini"] = f"✅ Working with model: {model_name}"
                        test_results["Gemini Response"] = response.text[:100]
                        break
                    except Exception as e:
                        continue
                if not working_model:
                    test_results["Gemini Error"] = "No compatible model found. Check your API key and quota."
            except Exception as e:
                test_results["Gemini Error"] = str(e)
        else:
            test_results["Gemini"] = "Not configured"
        
        # Test News API
        if NEWS_API_KEY:
            result = fetch_news_via_api("AI", page_size=1)
            if result.get("error"):
                test_results["News API Error"] = result["error"]
            else:
                test_results["News API"] = f"✅ Fetched {len(result.get('articles', []))} article(s)"
        else:
            test_results["News API"] = "Not configured"
        
        st.json(test_results)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"<div class='msg-user'><strong>You:</strong><div class='msg-text'>{content}</div></div>", 
                   unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-assistant'><strong>MAVI:</strong><div class='msg-text'>{content}</div></div>", 
                   unsafe_allow_html=True)

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message:", placeholder="Type a message or use /calc, /news, /echo commands...", height=100)
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        send_btn = st.form_submit_button("📤 Send", use_container_width=True)
    with col2:
        if st.form_submit_button("🔍 /news", use_container_width=True):
            user_input = "/news latest technology"
            send_btn = True
    with col3:
        if st.form_submit_button("🧮 /calc", use_container_width=True):
            user_input = "/calc 2+2"
            send_btn = True

if send_btn and user_input and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    if use_news and any(user_input.lower().startswith(kw) for kw in ["latest", "news", "show news", "get news"]):
        news_result = fetch_news_via_api(user_input, page_size=5)
        if news_result.get("error"):
            response = f"News fetch error: {news_result['error']}"
        else:
            articles = news_result.get("articles", [])
            response = "📰 Latest News:\n\n"
            for i, a in enumerate(articles, 1):
                response += f"{i}. **{a.get('title')}** ({a.get('source')})\n"
                response += f"   {a.get('publishedAt')} • {a.get('url')}\n\n"
            if use_gemini and GEMINI_READY:
                summary = summarize_articles_with_gemini(articles, user_input)
                response += f"\n**Summary:**\n{summary}"
    else:
        with st.spinner("MAVI is thinking..."):
            response = agent.handle_input(user_input, use_rag=use_rag, use_gemini_fallback=use_gemini)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()

# Image analysis
uploaded_image = st.file_uploader("📸 Upload image for analysis", type=["png", "jpg", "jpeg"]) 
if uploaded_image and st.button("🔍 Analyze Image"):
    img_bytes = uploaded_image.read()
    if use_gemini and GEMINI_READY:
        with st.spinner("Analyzing image with Gemini..."):
            result = analyze_image_with_gemini(img_bytes, "Describe this image in detail and extract any text.")
    else:
        ocr_text = extract_text_from_image_bytes(img_bytes)
        result = f"OCR Text Extraction:\n\n{ocr_text}" if ocr_text else "No text found in image."
    st.session_state.chat_history.append({"role": "user", "content": "[Image uploaded for analysis]"})
    st.session_state.chat_history.append({"role": "assistant", "content": result})
    st.rerun()

# Document upload for RAG
st.markdown("---")
st.subheader("📚 Document Knowledge Base")
uploaded_docs = st.file_uploader("Upload documents (PDF, TXT, Images)", 
                                 accept_multiple_files=True, 
                                 type=["pdf", "txt", "png", "jpg", "jpeg"]) 

if st.button("📥 Index Documents"):
    if not uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Processing documents..."):
            texts = []
            for doc in uploaded_docs:
                data = doc.read()
                text = ""
                if doc.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(data)
                elif any(doc.name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                    text = extract_text_from_image_bytes(data)
                else:
                    try:
                        text = data.decode("utf-8")
                    except Exception:
                        pass
                if text.strip():
                    texts.append(text)
            if texts:
                agent.add_documents(texts)
                st.success(f"✅ Indexed {len(texts)} documents!")
            else:
                st.error("No text could be extracted from uploaded files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
<strong>MAVI AI Agent</strong> • Local-first AI with cloud fallback • 
<a href='https://github.com' target='_blank'>Documentation</a>
</div>
""", unsafe_allow_html=True)
