
# 🤖 BookNess AI Chatbot

An advanced, production-grade AI assistant designed to streamline operations for the BookNess bookkeeping platform. The chatbot enables **admins** and **project managers** to interact with backend data using natural language queries — intelligently translating questions into SQL, generating reports, and handling conversational context.

## 🔍 Key Features

- 💬 **Natural Language Interface**  
  Understands business queries like _"Show me all completed projects by writers last month."_  

- 🧠 **Session-Based Memory**  
  Remembers conversation context during a session, with role-specific interactions (e.g., admin vs manager).  

- 🔐 **Secure SQL Generation**  
  Dynamically generates and validates SQL queries — blocks unsafe operations.  

- 📊 **Report Generation & Export**  
  Generates downloadable CSV reports from chat-based queries with structured filtering options.  

- 🕓 **Time-Based Reasoning**  
  Understands vague time phrases like _“last month”_, _“this quarter”_, _“previous year”_, etc.  

- 🎤 **(Optional) Voice Agent Support**  
  Extendable to support real-time voice assistants using LiveKit, Deepgram, and ElevenLabs.

- 🚀 **Built with FastAPI**  
  Backend-ready for deployment with streaming responses and modular architecture.

---

## 🛠 Tech Stack

- **LLMs**: OpenAI GPT-4 / Gemini Pro / GPT-4o
- **Backend**: Python, FastAPI
- **Database**: PostgreSQL / MySQL / MongoDB (customizable)
- **Memory**: In-memory session tracking (extendable to Redis)
- **Deployment**: Render / GCP / Docker
- **Extras**: Gradio, LiveKit, Deepgram, ElevenLabs (optional voice support)

---

## 📁 Project Structure

