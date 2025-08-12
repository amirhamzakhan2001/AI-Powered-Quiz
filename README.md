# AI-Powered Quiz for Rural Learners

A team project for the **IBM SkillBuild Agentic AI Certification Course**  
Empowering students with personalized, multilingual, AI-generated quizzes via RAG and LLMs.

## 📂 Folder Structure

```
AI-Powered-Quiz/
│
├── app.py
├── .env.example # ← provide template/example here, real .env in .gitignore
├── requirements.txt
├── README.md
├── backend/
│ ├── student_data.py
│ ├── rag_vector_store.py
│ ├── langgraph_workflow.py
│ ├── quiz_evaluation_graph.py
│ ├── performance_evaluator.py
│ ├── question_parser.py
│ └── __init__.py
```

## 📝 Project Overview

The goal is to provide rural and underserved learners with interactive, customized quizzes across grade levels and subjects. The app uses Retrieval-Augmented Generation (RAG) with Google Gemini LLM, an intuitive Streamlit interface, and efficient knowledge retrieval through vector search so that quizzes are tailored and actionable feedback is realistic. Data is persistently stored in MongoDB for longitudinal learner tracking.

## 🚀 Features

- Multilingual quiz generation (Grades 3–12, Bachelor’s, Master’s)
- Retrieval-Augmented Generation (RAG) to bring domain context into every quiz
- Auto-grading and personalized LLM-powered feedback
- Data persistence via MongoDB for tracking learner progress
- Accessible UI with Streamlit, suitable for low-bandwidth environments
- Team project for education/research use (IBM SkillBuild)

## ⚙️ Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/your-team/ai-powered-quiz.git
cd ai-powered-quiz
```

### 2. Install dependencies

Create and activate a Python 3.9+ virtual environment, then run:
```
pip install -r requirements.txt
```

### 3. Configure environment variables

- **Never commit your `.env` file!**  
- Copy `.env.example` to `.env` and add your real keys.

cp .env.example .env


Fill in your `.env` with:
```
GOOGLE_API_KEY=your-google-gemini-api-key
MONGODB_URI=your-mongodb-atlas-uri
```

Your `.env` is protected by `.gitignore`; only your API users with proper keys can run the app. If someone clones your repo, they must provide their own credentials.

### 4. Run locally with Streamlit
```
streamlit run app.py
```

The app will launch in your browser.

## 🔗 Environment Variables

| Variable       | Purpose                 | Example/Format        |
|----------------|------------------------|----------------------|
| GOOGLE_API_KEY | Google Gemini LLM access | `<your-gemini-key-here>` |
| MONGODB_URI    | MongoDB Atlas connection | `mongodb+srv://...`  |

## 💡 Tech Stack

- Python 3.9+
- Streamlit (UI)
- LangChain + LangGraph
- Google Gemini LLM
- MongoDB Atlas
- FAISS Vector Store
- dotenv (for environment config)

## 📊 Architecture Diagram

<details>
<summary>Expand to view architecture</summary>

[ Student UI / Streamlit ]
|
v
[ app.py ]
|
| | |
[ Quiz [ Quiz [ MongoDB
Generator ] Evaluation ] DataStore ]
| | |
[RAG/vector [ LLM [ Student
Search Report/FB ] Records ]
& Gemini ]


</details>

## 📷 Demo

- Add a screenshot or GIF below after your first successful run  
- If a YouTube video link is ready, insert it here.

## 👥 Team Acknowledgement

This project is a collaborative effort by a 6-member team as part of the **IBM SkillBuild – Agentic AI Certification**.  
For educational purposes only.

## 🛡️ License

This repository is licensed under the **MIT License** and is designed for educational and research use. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Open to positive, educational collaborations. Please create issues or PRs for any improvements or questions.

---
