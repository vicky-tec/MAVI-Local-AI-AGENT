# MAVI – Local AI Agent (Memory-Augmented Virtual Intelligence)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

**MAVI** is a **local-first AI assistant** that runs entirely on your machine using **Ollama** for local LLMs, with optional **Gemini** cloud fallback for vision and enhanced reasoning.  
Built for **students, developers, and privacy-focused users**, MAVI delivers RAG, OCR, modern tools, and smart file organization—while keeping all data on your device.

---

## 📋 Table of Contents

1. [Features](#-features)
2. [Architecture](#-architecture)
3. [Prerequisites](#-prerequisites)
4. [Installation (Step-by-Step)](#-installation-step-by-step)
5. [Configuration](#-configuration)
6. [How to Run](#-how-to-run)
7. [Usage Guide](#-usage-guide)
8. [Project Structure](#-project-structure)
9. [Roadmap / Future Work](#-roadmap--future-work)
10. [Troubleshooting](#-troubleshooting)
11. [License](#-license)

---

## 🚀 Features

- **🧠 Local Intelligence**  
  Runs on `llama3.2:3b` (via Ollama) for offline privacy-first inference.

- **🔎 RAG (Retrieval-Augmented Generation)**  
  Upload PDFs or images → MAVI extracts text and answers questions based on your docs.

- **🛠️ Smart Tools**
  - `/calc` — Safe calculator  
  - `/search` — DuckDuckGo web search  
  - `/news` — Real-time news  
  - `/fetch` — Scrape & clean webpage text  

- **👁 Vision Capabilities**  
  Gemini Vision describes, interprets, and analyzes images.

- **🗂 File Organizer**  
  Automatically organizes messy folders (Images, Videos, Documents, Code, etc.).

- **💾 Long-Term Memory**  
  Stores conversation history and document indexes in SQLite.

- **💻 Modern Streamlit UI**  
  Midnight-dark interface with clean chat experience.

---

## 🏗 Architecture

- **Backend:** Python + SQLite  
- **LLM Engine (Primary):** Ollama  
  - Chat model: `llama3.2:3b`  
  - Embedding model: `qwen3-embedding:0.6b`
- **Vision/Fallback Engine:** Google Gemini API  
- **Frontend:** Streamlit UI running on `localhost:8501`

---

## ⚙ Prerequisites

### System Requirements
- **OS:** Windows / macOS / Linux  
- **RAM:** 8GB minimum (16GB recommended)

### Software Required
- **Python 3.10+**
- **Ollama**  
- **Tesseract OCR** (required for extracting text from images)

Links:
- Python → https://www.python.org/downloads/  
- Ollama → https://ollama.com  
- Tesseract OCR → https://github.com/tesseract-ocr/tesseract  

---

## 📦 Installation (Step-by-Step)

Follow each step carefully:

---

### **Step 1: Install & Configure Ollama**

Install Ollama from:  
👉 https://ollama.com

Then pull required models:


* ollama pull llama3.2:3b
* ollama pull qwen3-embedding:0.6b
---
Verify:
> ollama list
-------
Step 2: Clone the Repository
> git clone https://github.com/vicky-tec/MAVI-Local-AI-AGENT.git
cd mavi-local-agent
-------
Step 3: Set up Python Environment
Windows
* python -m venv .venv
* .\.venv\Scripts\activate
-------
Mac/Linux
* python3 -m venv .venv
* source .venv/bin/activate
--------
Step 4: Install Dependencies
* pip install --upgrade pip
* pip install -r requirements.txt
----------
Step 4: Install Dependencies
* pip install --upgrade pip
* pip install -r requirements.txt
-----------
### 🔧 Configuration

Create a .env file in the project root:

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
LLM_MODEL=llama3.2:3b
EMBED_MODEL=qwen3-embedding:0.6b

# API Keys (Optional)
* GEMINI_API_KEY=your_gemini_key_here
* NEWS_API_KEY=your_newsapi_key_here</h4>  
-------------
### ▶️ How to Run
1. Run Diagnostic Tests (Optional)
python test_ollama.py
python gemini_test.py
-------------
2. Start MAVI
streamlit run mavi_unified.py
Open in browser:
👉 http://localhost:8501
--------------
### 🛠 Slash Commands
| Command   | Example                      | Description         |
| --------- | ---------------------------- | ------------------- |
| `/calc`   | `/calc (50*4)/2`             | Safe calculator     |
| `/search` | `/search AI news`            | DuckDuckGo search   |
| `/news`   | `/news technology`           | Latest headlines    |
| `/fetch`  | `/fetch https://example.com` | Scrape webpage text |
-----------------
### 📚 RAG (Document Chat)

Upload a PDF or Image
MAVI extracts text (PDF OCR + embeddings)

Ask:
* “Summarize the document I uploaded.”
* “What are the key points?”</h4>  
-------------------
### 🗂 File Organizer

Run in a Python shell:

* from file_organizer import organize
* organize("C:/Users/Downloads")</h4>  
----------------------
### 🗺 Roadmap / Future Work

* File Search Tool → /find <keyword>
* Document Summarizer → /summarize file.pdf
* CSV/Excel Data Analyzer
* YouTube Transcript Summaries
* OS Tools → App launcher, clipboard manager
* Voice Mode → Speech-to-text + TTS
* Multi-Agent System
------------------------------
❓ Troubleshooting</br>
Issue	Solution</br>
ollama not found	Install Ollama and restart terminal</br>
Cannot connect to Ollama	Ensure service is running: ollama list</br>
OCR error	Install Tesseract & configure PATH</br>
API key issues	Check .env formatting — no quotes</br>
Gemini errors	Run: python gemini_test.py</br>
**------------------**
📄 License

MIT License
FREE TO USE.

Created by **Vicky Raj & Team (Cosmic Shakti)**.
