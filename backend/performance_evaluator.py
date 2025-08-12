# backend/performance_evaluator.py
# used to generate personalized feedback, performance report, and evaluate performance

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize once at the module level for reuse
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in your environment.")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)



def extract_or_generate_subject_topic(question_text: str, class_selected: str, selected_subject, general_topics : list) -> tuple:
    """
    Decide subject/topic for a question.
    - In General mode: prompt with topics list.
    - In specific-subject mode: prompt with class+subject.
    """
    # Detect general mode: topics list provided
    is_general_mode = isinstance(selected_subject, list) or bool(general_topics)

    if is_general_mode:
        prompt = f"""
        You are given learning level: "{class_selected}".
        Topics: {selected_subject}
        Based on the following quiz question, identify and confirm subject and most relevant topic.
        Question: {question_text}
        Respond only as JSON: {{ "subject": "...", "topic": "..." }}
        """
    else:
        prompt = f""" you are given :
        Class: "{class_selected}"
        Subject: "{selected_subject}"
        Based on the following quiz question, identify the most appropriate topic within this subject.

        Question: {question_text}
        Respond only as JSON: {{ "subject": "{selected_subject}", "topic": "..." }}
        """

    try:
        response = llm.predict(prompt)
        start, end = response.find("{"), response.rfind("}") + 1
        result = json.loads(response[start:end])
        subject = result.get("subject", selected_subject if selected_subject else "Unknown")
        topic = result.get("topic", "Unknown")
    except Exception as e:
        st.error(f"Error parsing subject/topic from LLM response: {e}")
        subject, topic = (selected_subject if selected_subject else "Unknown"), "Unknown"

    return subject, topic
    

def evaluate_answers(questions: list, answers: list) -> list:
    """
    Compare user's answers with correct answers and return detailed evaluation.
    """
    results = []
    for i, q in enumerate(questions):
        user_answer = answers[i] if i < len(answers) else None
        correct_answer = q.get("correct")
        is_correct = (user_answer == correct_answer)
        results.append({
            "question": q["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "subject": q.get("subject", "Unknown"),
            "topic": q.get("topic", "Unknown"),
        })
    return results

def generate_performance_report(evaluation_results: list, language: str = "English") -> str:
    """
    Use Gemini LLM to generate a performance report in the specified language.
    """
    prompt = f"""
    Given quiz results (JSON): {json.dumps(evaluation_results, indent=2)}
    Give a clear and concise performance report in {language} including:
    - Total questions
    - Number of correct and incorrect answers
    - Breakdown of performance by subject and by topic
    
    Respond with a formatted textual summary.        
    """
    try:
        response = llm.predict(prompt)
        return response
    except Exception as e:
        st.error(f"Error generating performance report: {e}")
        return "Failed to generate performance report."


def generate_personalized_feedback(evaluation_results: list, language: str = "English") -> str:
    """
    Generate personalized, motivational, actionable feedback for the student 
    based on detailed quiz evaluation results using the Gemini LLM.
    """

    prompt = f"""
    You are an AI tutor analyzing a student's quiz performance.

    The quiz comprised {len(evaluation_results)} questions.

    Each question is characterized by the subject, topic, whether the student's answer was correct, and overall accuracy.

    Here are the quiz details (in JSON format):

    {json.dumps(evaluation_results, indent=2)}

    Provide a personalized, encouraging, and constructive feedback for the student, including:

    - Overall performance summary
    - Highlights of subjects and topics where the student excelled
    - Subjects and topics which need improvement
    - Specific advice on how to improve weaker areas
    - Motivation to keep learning and practicing

    Please respond in {language}.

    Give the feedback in 3-5 concise paragraphs.
    """

    try:
        response = llm.predict(prompt)
        return response
    except Exception as e:
        import streamlit as st
        st.error(f"Error generating personalized feedback: {e}")
        return "Failed to generate personalized feedback."

