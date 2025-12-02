# MAVI – Local AI Agent (Memory-Augmented Virtual Intelligence)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

**MAVI** is a **local-first AI assistant** that runs primarily on your own machine using **Ollama** (local LLMs) with optional **Gemini** cloud fallback. Designed for students, developers, and privacy-conscious users, MAVI keeps your data on your device while offering powerful tools like RAG, vision analysis, and smart file organization.

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
9. [Roadmap](#-roadmap--future-work)
10. [Troubleshooting](#-troubleshooting)

---

## 🚀 Features

* **🧠 Local Intelligence:** Runs on `llama3.2:3b` via Ollama for privacy and offline capability.
* **🔎 RAG (Retrieval-Augmented Generation):** Upload PDFs or images; MAVI indexes them and answers questions based on your documents.
* **🛠️ Smart Tools:**
    * `/calc`: Safe mathematical calculator.
    * `/search`: Web search via DuckDuckGo.
    * `/news`: Real-time news updates.
    * `/fetch`: Scrape and clean text from URLs.
* **👁️ Vision Capabilities:** Uses Gemini Vision to analyze and describe uploaded images.
* **🗂️ File Organizer:** Automatically sorts clutter in folders into Images, Documents, Code, etc.
* **💾 Long-term Memory:** Uses SQLite to store conversation history and document embeddings.
* **💻 Modern UI:** Built with Streamlit featuring a "Midnight" dark theme.

---

## 🏗 Architecture

* **Backend:** Python 3 + SQLite (Memory & Vector Store).
* **AI Engine (Primary):** Ollama (`llama3.2:3b` for chat, `qwen3-embedding:0.6b` for RAG).
* **AI Engine (Fallback/Vision):** Google Gemini.
* **Frontend:** Streamlit (Localhost).

---

## ⚙ Prerequisites

Before installing the Python code, ensure you have the following system requirements:

1.  **OS:** Windows, macOS, or Linux.
2.  **RAM:** 8GB minimum (16GB recommended).
3.  **Software:**
    * [Python 3.10+](https://www.python.org/downloads/)
    * [Ollama](https://ollama.com/)
    * [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Required for image text extraction).

---

## 📦 Installation (Step-by-Step)

Follow these steps exactly to get MAVI running.

### Step 1: Install & Configure Ollama
1.  Download and install **Ollama** from [ollama.com](https://ollama.com).
2.  Open your terminal/command prompt and pull the necessary models:
    ```bash
    ollama pull llama3.2:3b
    ollama pull qwen3-embedding:0.6b
    ```
3.  Verify Ollama is running by typing `ollama list`.

### Step 2: Clone the Repository
```bash
git clone [https://github.com/YourUsername/mavi-local-agent.git](https://github.com/YourUsername/mavi-local-agent.git)
cd mavi-local-agent
Step 3: Set up Python EnvironmentWindows:Bashpython -m venv .venv
.venv\Scripts\activate
Mac/Linux:Bashpython3 -m venv .venv
source .venv/bin/activate
Step 4: Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
Step 5: Install Tesseract OCRWindows: Download Installer. Important: Add Tesseract to your System PATH during installation.Mac: brew install tesseractLinux: sudo apt-get install tesseract-ocr🔧 ConfigurationMAVI uses environment variables to manage API keys.Create a file named .env in the root folder.Copy and paste the following into it:Ini, TOML# .env file

# Ollama Settings (Local)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
LLM_MODEL=llama3.2:3b
EMBED_MODEL=qwen3-embedding:0.6b

# API Keys (Optional but recommended)
# Get key: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_key_here

# Get key: [https://newsapi.org/](https://newsapi.org/)
NEWS_API_KEY=your_newsapi_key_here
▶️ How to Run1. Diagnostic Test (Optional)Run these scripts to ensure your environment is ready.Bash# Test local LLM connection
python test_ollama.py 

# Test Cloud/Vision connection
python gemini_test.py
2. Start MAVIRun the main application:Bashstreamlit run mavi_unified.py
A browser window will automatically open at http://localhost:8501.📖 Usage GuideBasic ChatSimply type in the chat box to talk to the local AI."Explain Quantum Computing.""Write a Python script to sort a list."Slash CommandsMAVI supports specific tools triggered by / commands:CommandUsageDescription/calc/calc (50*4)/2Solves math expressions safely./search/search AI newsSearches the web using DuckDuckGo./news/news technologyFetches live top headlines via NewsAPI./fetch/fetch https://...Scrapes text from a specific URL.RAG (Document Chat)Use the sidebar to Upload a PDF or Image.MAVI will process and index the file.Ask questions: "Summarize the document I just uploaded."File OrganizerTo organize a cluttered folder (e.g., Downloads), you can use the built-in script:Python# In a separate python script or terminal
from file_organizer import organize
organize("C:/Users/YourName/Downloads")
🗺 Roadmap / Future WorkWe are actively working on these features:[ ] File Search: /find command to search local files and content.[ ] Summarization: /summarize for large local documents.[ ] Data Analysis: CSV/Excel visualization and statistics.[ ] Media Tools: YouTube video transcript extraction and summary.[ ] OS Integration: App launcher (/open code) and clipboard tools.[ ] Voice Mode: Speech-to-text and Text-to-Speech (TTS) integration.📂 Project StructureBash.
├── mavi_unified.py        # 🚀 MAIN APP entry point
├── requirements.txt       # Dependencies
├── .env                   # API Keys & Config (Create this!)
├── file_organizer.py      # File sorting logic
├── test_ollama.py         # Diagnostic tool for Ollama
├── gemini_test.py         # Diagnostic tool for Gemini
└── mavi_unified.db        # Database (Auto-created)
❓ TroubleshootingIssueSolutionollama not foundEnsure Ollama is installed and added to your System PATH.Connection ErrorCheck if Ollama is running (ollama list) and port 11434 is open.OCR/Tesseract ErrorOn Windows, ensure Tesseract is in PATH. On Linux/Mac, verify installation.API ErrorsCheck your .env file to ensure API keys are pasted correctly without quotes.📄 LicenseDistributed under the MIT License. See LICENSE for more information.Original project by Vicky Raj and Team (Cosmic Shakti).
### Next Steps I can do for you:

Would you like me to write the content for the **`test_ollama.py`** or **`gemini_test.py`** scripts so you can include them in the repo immediately?

<h1> FREE TO USE </h1>
