# 🛡️ AI Governance & Prompt Evaluation Framework

An AI governance framework designed to evaluate, test, and monitor Large Language Model (LLM) applications. The project combines prompt evaluation, Retrieval-Augmented Generation (RAG), governance checks, memory management, debugging tools, and automated reporting to help build reliable and trustworthy AI systems.

---

# ✨ Features

- 🤖 AI prompt evaluation
- 📊 Automated evaluation reports
- 📚 Retrieval-Augmented Generation (RAG)
- 🛡️ AI governance and policy validation
- 🧠 Memory management
- 🧪 A/B testing support
- 🔍 Prompt debugging utilities
- 📈 Evaluation dashboard
- 📄 Governance report generation
- ⚡ Modular Python architecture

---

# 🏗️ Architecture

```text
               User Prompt
                    │
                    ▼
           Prompt Evaluation Layer
                    │
                    ▼
         Governance & Safety Checks
                    │
                    ▼
        RAG Retrieval + Memory Module
                    │
                    ▼
          LLM Response Generation
                    │
                    ▼
      Evaluation & Governance Report
```

---

# 📁 Project Structure

```text
AI-Governance/
│
├── app.py
├── ingest.py
├── retrieve.py
├── evaluate.py
├── governance.py
├── memory.py
├── tools.py
├── generate.py
├── promptfoo_wrapper.py
├── ab_testing.py
├── governance_report.json
├── evaluation_results.json
├── requirements.txt
├── README.md
└── chroma_db/
```

---

# 🛠️ Tech Stack

### Backend
- Python

### AI & RAG
- OpenAI / OpenRouter
- LangChain
- ChromaDB

### Evaluation
- Promptfoo
- JSON Reports
- A/B Testing

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Jyothi-D3/<repository-name>.git

cd <repository-name>
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment

Create a `.env` file.

```env
OPENAI_API_KEY=your_api_key
```

## Run

```bash
python app.py
```

---

# 🔄 Workflow

1. Load the knowledge base.
2. Retrieve relevant context using RAG.
3. Execute governance and safety checks.
4. Generate an AI response.
5. Evaluate response quality.
6. Run A/B testing (optional).
7. Produce governance and evaluation reports.

---

# 🎯 Capabilities

- Prompt Evaluation
- AI Governance
- Safety Validation
- Memory Management
- RAG-based Retrieval
- Automated Reporting
- Prompt Debugging
- A/B Testing

---

# 📊 Reports

The project generates:

- Evaluation Reports
- Governance Reports
- Performance Metrics
- Prompt Analysis
- Safety Checks

---

# 📷 Screenshots

```text
screenshots/
├── dashboard.png
├── evaluation.png
├── governance.png
└── reports.png
```

Example:

```markdown
## Dashboard

![Dashboard](screenshots/dashboard.png)
```

---

# 🚀 Future Enhancements

- Multi-model evaluation
- Bias detection
- Hallucination detection
- Cost and latency tracking
- CI/CD integration
- Real-time monitoring
- Interactive governance dashboard

---

# 👩‍💻 Author

**Jyothi Islavath**

GitHub: https://github.com/Jyothi-D3

---

# 📄 License

This project is developed for educational and portfolio purposes.
