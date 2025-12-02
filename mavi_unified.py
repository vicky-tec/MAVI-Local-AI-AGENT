"""
MAVI Unified AI Agent - Production Ready
Combines the best features from both mavi.py and stream.py
Now includes: Intelligent File Organizer & Smart Real-time Search & Midnight Indigo UI

Features:
- Ollama: llama3.2:3b (chat) + qwen3-embedding:0.6b (embeddings)
- Gemini: gemini-2.5-pro (vision & fallback)
- RAG: Document Q&A with embeddings
- Tools: Calculator, Web Search, News, URL Fetch
- Smart Search: Random topic generation + News API integration
- OCR: Image text extraction
- Vision: Gemini image analysis
- UI: Premium "Midnight Indigo" Design with Glassmorphism
- File Organizer: Categorizes files in specific directories

Install:
pip install streamlit openai google-generativeai duckduckgo-search beautifulsoup4 requests PyPDF2 numpy python-dotenv pillow pytesseract
"""

import os
import io
import shutil
import sqlite3
import logging
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import streamlit as st
import numpy as np
from PIL import Image
import PyPDF2

# Optional imports with graceful fallback
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except:
    BS4_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except:
    GENAI_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except:
    SEARCH_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except:
    PYTESSERACT_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ .env loaded")
except Exception as e:
    logger.warning(f"⚠️ .env error: {e}")


def get_secret(key: str, default: Any = None) -> Any:
    """Get secret from env or Streamlit secrets"""
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        if key in st.secrets:
            val = st.secrets[key]
            return val.strip() if isinstance(val, str) else val
    except:
        pass
    return default


class Config:
    # API Keys
    GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
    NEWS_API_KEY = get_secret("NEWS_API_KEY")

    # Ollama
    OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = get_secret("OLLAMA_API_KEY", "ollama")
    LLM_MODEL = get_secret("LLM_MODEL", "llama3.2:3b")
    EMBED_MODEL = get_secret("EMBED_MODEL", "qwen3-embedding:0.6b")

    # Gemini
    GEMINI_MODEL = "gemini-2.5-pro" # Updated to latest flash or pro as needed

    # Storage
    DB_PATH = "mavi_unified.db"

    # UI Colors - Midnight Indigo Theme
    COLORS = {
        'bg_dark': '#0f172a',       # Deep Slate (Main Background)
        'bg_light': '#1e293b',      # Lighter Slate (Sidebar)
        'primary': '#6366f1',       # Electric Indigo
        'secondary': '#8b5cf6',     # Vivid Violet
        'accent': '#38bdf8',        # Sky Blue Glow
        'card_bg': 'rgba(30, 41, 59, 0.7)', # Glass effect for cards
        'user_bubble': 'linear-gradient(135deg, #3b82f6, #2563eb)', # Royal Blue Gradient
        'bot_bubble': '#334155',    # Slate Grey
        'success': '#10b981',       # Emerald
        'error': '#ef4444',         # Red
        'text_main': '#f8fafc',     # Off-white
        'text_sub': '#94a3b8',      # Muted grey
        'border': 'rgba(255, 255, 255, 0.1)'
    }


# Initialize Ollama
OLLAMA_CLIENT = None
if OPENAI_AVAILABLE:
    try:
        OLLAMA_CLIENT = OpenAI(
            base_url=Config.OLLAMA_BASE_URL,
            api_key=Config.OLLAMA_API_KEY
        )
        logger.info("✅ Ollama initialized")
    except Exception as e:
        logger.error(f"❌ Ollama error: {e}")

# Initialize Gemini
GEMINI_READY = False
if GENAI_AVAILABLE and Config.GEMINI_API_KEY:
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        GEMINI_READY = True
        logger.info("✅ Gemini initialized")
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")

# ============================================================================
# FILE ORGANIZER MODULE
# ============================================================================

class FileOrganizer:
    """Enhanced File Organizer Module"""
    
    EXTENSIONS = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff'],
        'Videos': ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'],
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xlsx', '.pptx', '.csv'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.json', '.xml', '.sql'],
        'Executables': ['.exe', '.msi', '.apk', '.dmg', '.deb']
    }

    @staticmethod
    def organize_folder(path: str) -> str:
        """Organizes files in the specified path"""
        target_path = Path(path)
        
        # Safety Checks
        if not target_path.exists():
            return "❌ Error: Path does not exist."
        
        # Prevent running on system roots to avoid OS damage
        forbidden_paths = [Path('/'), Path('C:\\'), Path('C:/'), Path('/bin'), Path('/usr'), Path('/Windows')]
        if target_path.resolve() in [p.resolve() for p in forbidden_paths if p.exists()]:
            return "❌ Safety Error: Cannot run on root or system directories."

        stats = {k: 0 for k in FileOrganizer.EXTENSIONS.keys()}
        stats['Others'] = 0
        
        try:
            for item in target_path.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    file_ext = item.suffix.lower()
                    moved = False
                    
                    for category, extensions in FileOrganizer.EXTENSIONS.items():
                        if file_ext in extensions:
                            dest_dir = target_path / category
                            dest_dir.mkdir(exist_ok=True)
                            
                            # Handle duplicates
                            dest_file = dest_dir / item.name
                            counter = 1
                            while dest_file.exists():
                                dest_file = dest_dir / f"{item.stem}_{counter}{item.suffix}"
                                counter += 1
                                
                            shutil.move(str(item), str(dest_file))
                            stats[category] += 1
                            moved = True
                            break
                    
                    if not moved:
                        dest_dir = target_path / "Others"
                        dest_dir.mkdir(exist_ok=True)
                        dest_file = dest_dir / item.name
                        shutil.move(str(item), str(dest_file))
                        stats['Others'] += 1
                        
            # Format report
            report = [f"✅ Organization Complete in: `{path}`"]
            for cat, count in stats.items():
                if count > 0:
                    report.append(f"- **{cat}**: {count} files")
            
            return "\n".join(report) if len(report) > 1 else "ℹ️ No files needed organizing."
            
        except Exception as e:
            return f"❌ Error during organization: {str(e)}"

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text()
                if t:
                    pages.append(t)
            except:
                pass
        return "\n\n".join(pages)
    except:
        return ""


def extract_text_from_image_bytes(b: bytes) -> str:
    """Extract text from image via OCR"""
    if not PYTESSERACT_AVAILABLE:
        return "[OCR not available - install pytesseract]"
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception as e:
        return f"[OCR error: {e}]"

# ============================================================================
# EMBEDDINGS & RAG
# ============================================================================


def call_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings using Ollama"""
    if not OLLAMA_CLIENT:
        return []
    try:
        res = OLLAMA_CLIENT.embeddings.create(
            model=Config.EMBED_MODEL, input=texts)
        embeds = []
        for item in res.data:
            emb = getattr(item, "embedding", None)
            if emb is not None:
                embeds.append(np.array(emb, dtype=np.float32))
        return embeds
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve_top_k(query: str, texts: List[str], embeds: List[np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve top-k similar documents"""
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
# TOOLS
# ============================================================================


class ToolBox:
    """All agent tools"""

    @staticmethod
    def calculator(expression: str) -> str:
        """Calculator - /calc <expression>"""
        try:
            expr = re.sub(r'/calc\s*', '', expression.lower()).strip()
            allowed = set("0123456789+-*/().% ")
            if not set(expr).issubset(allowed):
                return "❌ Invalid characters. Use: 0-9 + - * / ( ) . %"
            result = eval(expr, {"__builtins__": {}})
            return f"### 🧮 Calculator\n**Expression:** `{expr}`\n**Result:** `{result}`"
        except Exception as e:
            return f"❌ Error: {e}"

    @staticmethod
    def web_search(query: str) -> str:
        """Web search - /search <query>"""
        if not SEARCH_AVAILABLE:
            return "⚠️ Install: pip install duckduckgo-search"
        try:
            clean_query = re.sub(r'/search\s*', '', query.lower()).strip()
            if not clean_query:
                return "⚠️ Provide a search query"

            with DDGS() as ddgs:
                results = list(ddgs.text(clean_query, max_results=5))

            if not results:
                return f"ℹ️ No results for: '{clean_query}'"

            output = f"### 🔍 Search Results: {clean_query}\n\n"
            for i, r in enumerate(results, 1):
                output += f"**{i}. [{r['title']}]({r['href']})**\n"
                output += f"   {r['body'][:200]}...\n\n"

            return output
        except Exception as e:
            return f"❌ Search error: {e}"

    @staticmethod
    def get_news(query: str = "") -> str:
        """News fetcher - /news <topic>"""
        if not Config.NEWS_API_KEY:
            # Fallback to web search if News API is missing
            return ToolBox.web_search(f"{query} news")
            
        try:
            topic = re.sub(r'/news\s*', '', query.lower()).strip() or "technology"

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": topic,
                "apiKey": Config.NEWS_API_KEY.strip(),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5
            }

            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 401:
                return "❌ Invalid NEWS_API_KEY"
            elif resp.status_code != 200:
                # Fallback on HTTP error
                return ToolBox.web_search(f"{topic} news")

            data = resp.json()
            articles = data.get('articles', [])

            if not articles:
                return f"ℹ️ No news for: '{topic}'. Try /search instead."

            output = f"### 📰 Latest News: {topic.title()} (Real-time)\n\n"
            for i, a in enumerate(articles, 1):
                title = a.get('title', 'No title')
                url = a.get('url', '#')
                source = a.get('source', {}).get('name', 'Unknown')
                desc = a.get('description', '')
                output += f"**{i}. [{title}]({url})**\n"
                output += f"   *Source: {source}*\n"
                if desc:
                    output += f"   > {desc[:150]}...\n\n"

            return output
        except Exception as e:
            return f"❌ News error: {e}"

    @staticmethod
    def fetch_url(url: str) -> str:
        """Fetch webpage - /fetch <url>"""
        if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            return "⚠️ Install: requests beautifulsoup4"
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            soup = BeautifulSoup(response.content, 'html.parser')

            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(
                chunk for line in lines for chunk in line.split("  ") if chunk)

            return f"### 🌐 Page Content\n{text[:2000]}..."
        except Exception as e:
            return f"❌ Fetch error: {e}"

# ============================================================================
# GEMINI FUNCTIONS
# ============================================================================


def analyze_image_with_gemini(image_bytes: bytes, prompt: str = "Describe this image") -> str:
    """Analyze image with Gemini Vision"""
    if not GEMINI_READY:
        return "⚠️ Gemini not configured"
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"❌ Gemini vision error: {e}"


def call_gemini_text(prompt: str, max_tokens: int = 800) -> str:
    """Generate text with Gemini"""
    if not GEMINI_READY:
        return "⚠️ Gemini not configured"
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens)
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# ============================================================================
# MEMORY MANAGER
# ============================================================================


class MemoryManager:
    """SQLite-based memory"""

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                model TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_message(self, role: str, content: str, model: str = ""):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO conversations (role, content, model) VALUES (?, ?, ?)",
            (role, content, model)
        )
        conn.commit()
        conn.close()

    def get_messages(self, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT role, content, model FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()
        return [
            {"role": r[0], "content": r[1], "model": r[2]}
            for r in reversed(rows)
        ]

    def clear_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()

    def add_document(self, filename: str, content: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (filename, content) VALUES (?, ?)",
            (filename, content)
        )
        conn.commit()
        conn.close()

    def get_all_documents(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT content FROM documents")
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]

# ============================================================================
# AGENT ENGINE
# ============================================================================


class MAVIAgent:
    """Main AI agent"""

    def __init__(self, model_choice: str = "ollama"):
        self.model_choice = model_choice
        self.memory = MemoryManager()
        self.tools = ToolBox()

        # RAG corpus
        self.corpus_texts: List[str] = []
        self.corpus_embeds: List[np.ndarray] = []

        # Load existing documents
        self._load_documents()

    def _load_documents(self):
        """Load indexed documents into RAG"""
        docs = self.memory.get_all_documents()
        if docs:
            logger.info(f"Loading {len(docs)} documents into RAG...")
            embeds = call_embeddings(docs)
            if embeds:
                self.corpus_texts = docs
                self.corpus_embeds = embeds
                logger.info(f"✅ Loaded {len(embeds)} document embeddings")

    def add_documents(self, texts: List[str]):
        """Add documents to RAG"""
        if not texts:
            return 0
        embeds = call_embeddings(texts)
        if embeds:
            self.corpus_texts.extend(texts)
            self.corpus_embeds.extend(embeds)
            return len(embeds)
        return 0

    def generate_response(self, prompt: str, use_rag: bool = True) -> str:
        """Generate AI response with Smart Routing"""

        # ----------------------------------------------------
        # SMART SEARCH ROUTING LOGIC
        # ----------------------------------------------------
        # If user types ONLY "/search", pick a random trending topic & use News API
        if prompt.strip().lower() == '/search':
            tech_topics = [
                "latest AI news", 
                "cybersecurity updates", 
                "space exploration news", 
                "quantum computing breakthroughs",
                "latest smartphone technology",
                "renewable energy news",
                "robotics advancements"
            ]
            random_topic = random.choice(tech_topics)
            # Route to News API for real-time data on this topic
            return self.tools.get_news(random_topic)

        # ----------------------------------------------------
        # STANDARD TOOLS
        # ----------------------------------------------------
        if prompt.lower().startswith('/calc'):
            return self.tools.calculator(prompt)
        elif prompt.lower().startswith('/search'):
            return self.tools.web_search(prompt)
        elif prompt.lower().startswith('/news'):
            return self.tools.get_news(prompt)
        elif prompt.lower().startswith('/fetch'):
            url = prompt.split(None, 1)[1] if len(prompt.split()) > 1 else ""
            return self.tools.fetch_url(url)

        # RAG context
        context = ""
        if use_rag and self.corpus_texts:
            retrieved = retrieve_top_k(
                prompt, self.corpus_texts, self.corpus_embeds, k=3)
            if retrieved:
                context_parts = [f"[Doc {i+1} | score={s:.2f}]:\n{t[:600]}"
                                 for i, (t, s) in enumerate(retrieved)]
                context = "\n\n".join(context_parts)

        # Build prompt
        history = self.memory.get_messages(limit=5)
        memory_context = "\n".join(
            [f"{m['role']}: {m['content'][:150]}" for m in history[-3:]])

        full_prompt = f"""You are MAVI, a helpful AI assistant with access to tools.

Available commands:
- /calc <expr> - Calculator
- /search <query> - Web search (Smart mode active)
- /news <topic> - Latest news (Real-time)
- /fetch <url> - Get webpage

{f"Relevant documents:{context}" if context else ""}

{f"Previous context:{memory_context}" if memory_context else ""}

User: {prompt}

Provide a helpful, accurate response."""

        # Generate with selected model
        if self.model_choice == "ollama" and OLLAMA_CLIENT:
            try:
                response = OLLAMA_CLIENT.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"❌ Ollama error: {e}"

        elif self.model_choice == "gemini" and GEMINI_READY:
            try:
                return call_gemini_text(full_prompt, max_tokens=1000)
            except Exception as e:
                return f"❌ Gemini error: {e}"

        return "❌ No AI model available"

    def test_systems(self) -> Dict[str, str]:
        """Test all systems"""
        status = {}

        # Test Ollama
        if OLLAMA_CLIENT:
            try:
                resp = OLLAMA_CLIENT.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                status["Ollama Chat"] = f"✅ {Config.LLM_MODEL}"
            except Exception as e:
                status["Ollama Chat"] = f"❌ {str(e)[:50]}"

            try:
                emb = call_embeddings(["test"])
                if emb:
                    status["Ollama Embeddings"] = f"✅ {Config.EMBED_MODEL} ({len(emb[0])}d)"
                else:
                    status["Ollama Embeddings"] = "❌ Failed"
            except Exception as e:
                status["Ollama Embeddings"] = f"❌ {str(e)[:50]}"
        else:
            status["Ollama"] = "❌ Not running"

        # Test Gemini
        if GEMINI_READY:
            try:
                model = genai.GenerativeModel(Config.GEMINI_MODEL)
                resp = model.generate_content("Test")
                status["Gemini"] = f"✅ {Config.GEMINI_MODEL}"
            except Exception as e:
                status["Gemini"] = f"❌ {str(e)[:50]}"
        else:
            status["Gemini"] = "❌ Not configured"

        # Test Search
        if SEARCH_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    list(ddgs.text("test", max_results=1))
                status["Web Search"] = "✅ DuckDuckGo"
            except Exception as e:
                status["Web Search"] = f"❌ {str(e)[:50]}"
        else:
            status["Web Search"] = "❌ Not installed"

        # Test News
        if Config.NEWS_API_KEY and REQUESTS_AVAILABLE:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/top-headlines",
                    params={"country": "us", "pageSize": 1,
                            "apiKey": Config.NEWS_API_KEY.strip()},
                    timeout=5
                )
                status[
                    "News API"] = "✅ Connected" if resp.status_code == 200 else f"❌ HTTP {resp.status_code}"
            except Exception as e:
                status["News API"] = f"❌ {str(e)[:50]}"
        else:
            status["News API"] = "❌ Not configured"

        return status

# ============================================================================
# STREAMLIT UI WITH MIDNIGHT INDIGO THEME
# ============================================================================


def inject_css():
    """Inject custom CSS with Rich UI styling"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Main App Background */
        .stApp {{
            background: {Config.COLORS['bg_dark']};
            color: {Config.COLORS['text_main']};
            font-family: 'Inter', sans-serif;
        }}
        
        /* MAVI Header - Glassmorphism & Gradient */
        .mavi-header {{
            background: linear-gradient(135deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
            padding: 2.5rem;
            border-radius: 24px;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 50px -12px rgba(99, 102, 241, 0.5); /* Indigo Glow */
            border: 1px solid {Config.COLORS['border']};
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }}
        
        /* Subtle shine effect on header */
        .mavi-header::before {{
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 50%; height: 100%;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.1), transparent);
            transform: skewX(-25deg);
            animation: shine 6s infinite;
        }}
        
        @keyframes shine {{
            0% {{ left: -100%; }}
            20% {{ left: 200%; }}
            100% {{ left: 200%; }}
        }}
        
        .mavi-title {{
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin: 0;
            letter-spacing: -1px;
            text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}
        
        /* Chat Messages */
        .chat-message {{
            padding: 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            max-width: 85%;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            border: 1px solid {Config.COLORS['border']};
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .user-msg {{
            background: {Config.COLORS['user_bubble']};
            margin-left: auto;
            color: white;
            border-bottom-right-radius: 4px;
        }}
        
        .bot-msg {{
            background: {Config.COLORS['bot_bubble']};
            color: {Config.COLORS['text_main']};
            border-bottom-left-radius: 4px;
        }}
        
        .msg-label {{
            font-size: 0.75rem;
            opacity: 0.8;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: rgba(255,255,255,0.8);
        }}
        
        /* Stylish Buttons */
        .stButton > button {{
            background: {Config.COLORS['card_bg']};
            color: {Config.COLORS['text_main']};
            border: 1px solid {Config.COLORS['border']};
            border-radius: 16px;
            padding: 0.85rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            backdrop-filter: blur(5px);
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
            transform: translateY(-3px);
            border-color: rgba(255,255,255,0.3);
            box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.6);
            color: white;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: {Config.COLORS['bg_light']};
            border-right: 1px solid {Config.COLORS['border']};
        }}
        
        /* Status Badges */
        .status-badge {{
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin: 0.3rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        
        .badge-success {{ background: {Config.COLORS['success']}; color: #064e3b; }}
        .badge-error {{ background: {Config.COLORS['error']}; color: #441111; }}
        
        /* Input Fields Styling */
        .stTextInput > div > div > input {{
            background-color: {Config.COLORS['bg_light']};
            color: white;
            border: 1px solid {Config.COLORS['border']};
            border-radius: 12px;
            padding: 10px;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {Config.COLORS['primary']};
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }}
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    st.set_page_config(
        page_title="MAVI LOCAL AI Agent",
        page_icon="🤖",
        layout="wide"
    )

    inject_css()

    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = MAVIAgent("ollama")
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "ollama"

    agent = st.session_state.agent

    # Header with new styling
    st.markdown("""
    <div class="mavi-header">
        <h1 class="mavi-title">🤖 MAVI LOCAL</h1>
        <p style="color: white; opacity: 0.8; font-size: 1.1rem; margin-top: 10px;">
            <span class="status-badge badge-success">Online</span>
            Local AI Agent | Ollama + Gemini | RAG + Tools
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        # Model Selection
        st.markdown("#### 🧠 AI Mode")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Ollama", use_container_width=True):
                st.session_state.model_choice = "ollama"
                st.session_state.agent.model_choice = "ollama"
                st.rerun()

        with col2:
            if st.button("☁️ Gemini", use_container_width=True):
                st.session_state.model_choice = "gemini"
                st.session_state.agent.model_choice = "gemini"
                st.rerun()

        # Dynamic Status Indicator (New Styled Box)
        if st.session_state.model_choice == "ollama":
            current_model = "🏠 Ollama (llama3.2:3b)"
        else:
            current_model = "☁️ Gemini 2.5 Pro"
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; margin-top: 15px; border: 1px solid {Config.COLORS['border']};">
            <small style="color: {Config.COLORS['text_sub']};">Active Model</small>
            <div style="font-weight: 600; color: {Config.COLORS['text_main']}; margin-top: 5px;">
                {current_model}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ==================================================
        # FILE ORGANIZER SECTION
        # ==================================================
        st.markdown("#### 📂 File Organizer")
        st.caption("Organize specific folders (e.g., Downloads) by file type.")
        
        org_path = st.text_input("Folder Path:", placeholder="C:/Users/Name/Downloads")
        
        if st.button("🧹 Organize Files", use_container_width=True):
            if not org_path:
                st.warning("⚠️ Please enter a folder path.")
            else:
                with st.spinner("Organizing files..."):
                    result = FileOrganizer.organize_folder(org_path)
                    st.markdown(result)
        
        st.markdown("---")

        # RAG Status
        st.markdown("#### 📚 RAG Status")
        doc_count = len(agent.corpus_texts)
        if doc_count > 0:
            st.success(f"✅ {doc_count} documents indexed")
        else:
            st.warning("⚠️ No documents indexed")

        st.markdown("---")

        # System Test
        st.markdown("#### 🧪 System Test")
        if st.button("Test All Systems", use_container_width=True):
            with st.spinner("Testing..."):
                status = agent.test_systems()
                for name, result in status.items():
                    badge_class = "badge-success" if "✅" in result else "badge-error"
                    st.markdown(
                        f'<div class="status-badge {badge_class}">{name}: {result}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tools Guide
        st.markdown("#### 🛠️ Tools")
        st.markdown("""
        - `/calc 2+2*3` - Calculator
        - `/search` - Smart Auto-Search
        - `/search <query>` - Web Search
        - `/news <topic>` - Latest News
        - `/fetch <url>` - Get Page
        """)

        st.markdown("---")

        # Document Upload
        st.markdown("#### 📄 Upload Documents")
        uploaded_docs = st.file_uploader("PDF, TXT for RAG", type=[
                                         'pdf', 'txt'], accept_multiple_files=True)
        if uploaded_docs and st.button("📥 Index Documents"):
            with st.spinner("Processing..."):
                texts = []
                for doc in uploaded_docs:
                    try:
                        data = doc.read()
                        if doc.name.endswith('.pdf'):
                            text = extract_text_from_pdf_bytes(data)
                        else:
                            text = data.decode('utf-8', errors='ignore')

                        if text.strip():
                            texts.append(text)
                            agent.memory.add_document(doc.name, text)
                    except Exception as e:
                        st.error(f"Error with {doc.name}: {e}")

                if texts:
                    count = agent.add_documents(texts)
                    st.success(f"✅ Indexed {count} documents!")
                else:
                    st.error("No text extracted")

        st.markdown("---")

        # Image Upload
        st.markdown("#### 🖼️ Upload Image")
        uploaded_image = st.file_uploader(
            "PNG, JPG for analysis", type=['png', 'jpg', 'jpeg'])
        if uploaded_image and st.button("🔍 Analyze Image"):
            img_bytes = uploaded_image.read()

            with st.spinner("Analyzing..."):
                if GEMINI_READY:
                    result = analyze_image_with_gemini(
                        img_bytes, "Describe this image in detail and extract any text.")
                else:
                    result = extract_text_from_image_bytes(img_bytes)
                    result = f"### OCR Result\n{result}" if result else "No text found"

            agent.memory.add_message("user", "[Image uploaded for analysis]")
            agent.memory.add_message("assistant", result, "gemini-vision")
            st.rerun()

    # Chat Display
    messages = agent.memory.get_messages(limit=100)

    for msg in messages:
        role_label = "You" if msg['role'] == 'user' else "MAVI"
        msg_class = "user-msg" if msg['role'] == 'user' else "bot-msg"
        model_tag = f" ({msg['model']})" if msg.get('model') else ""

        st.markdown(f"""
        <div class="chat-message {msg_class}">
            <div class="msg-label">{role_label}{model_tag}</div>
            <div>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==================================================
    # QUICK ACTIONS UI (ENHANCED)
    # ==================================================
    st.markdown("---")
    
    # Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍 Search (AI)", use_container_width=True):
            st.session_state.quick_input = "/search latest AI news"
    with col2:
        if st.button("📰 Tech News", use_container_width=True):
            st.session_state.quick_input = "/news technology"
    with col3:
        if st.button("🧮 Calculator", use_container_width=True):
            st.session_state.quick_input = "/calc (125*45)/10"
    with col4:
        if st.button("💡 Help", use_container_width=True):
            st.session_state.quick_input = "What can you do?"
            
    # Row 2 (Extra Buttons)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        if st.button("🎲 Surprise Me", use_container_width=True):
            st.session_state.quick_input = "/search"
    with col6:
        if st.button("📝 Summarize", use_container_width=True):
            st.session_state.quick_input = "Summarize the last 3 messages."
    with col7:
        if st.button("🧪 System Test", use_container_width=True):
             # This triggers the test function directly
            st.session_state.quick_input = "Running system diagnostics..."
            with st.spinner("Testing..."):
                status = agent.test_systems()
                formatted_status = "\n".join([f"- **{k}**: {v}" for k,v in status.items()])
                agent.memory.add_message("assistant", f"### 🧪 System Diagnostics\n{formatted_status}")
                st.rerun()
    with col8:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            agent.memory.clear_history()
            st.rerun()

    # Chat Input
    user_input = st.chat_input("Ask MAVI or use /calc /search /news /fetch")

    # Handle quick input
    if "quick_input" in st.session_state:
        user_input = st.session_state.quick_input
        del st.session_state.quick_input

    # Process input
    if user_input:
        agent.memory.add_message("user", user_input)

        current = "🏠 Ollama" if st.session_state.model_choice == "ollama" else "☁️ Gemini"
        with st.spinner(f"MAVI ({current}) thinking..."):
            response = agent.generate_response(user_input, use_rag=True)

        agent.memory.add_message(
            "assistant", response, st.session_state.model_choice)
        st.rerun()


if __name__ == "__main__":
    main()