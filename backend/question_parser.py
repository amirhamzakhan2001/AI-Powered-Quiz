# backend/quiz_core.py

import os
import re
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_question_generator_chain(rag_context: Optional[str] = None):
    
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Google Gemini API Key not found. "
            "Please set the GOOGLE_API_KEY environment variable "
            "or add it to a .env file in the root directory."
        )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)
    
    
    base_template = """
    You are an AI quiz generator. Generate {n} multiple-choice questions for a student of {class_name}.
    Topic(s): {subject_str}
    """
    if rag_context:
        base_template += f"\nUse this context: {rag_context}\n"
    
    base_template += """
    Each question must have exactly four options (A, B, C, D) and one correct answer.
    The quiz must be in Language: {language}

    Format:
    1. Question: ...
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: A
    """
    prompt_template = PromptTemplate(
        input_variables=["n", "class_name", "subject_str", "language"],
        template=base_template
    )


    chain = prompt_template | llm | StrOutputParser()
    return chain


def generate_questions_with_langchain(n, class_name, subject, language, rag_context: Optional[str] = None):
    """
    Generates quiz questions using the LangChain-integrated Gemini model,
    optionally augmented with RAG context.
    """
    subject_str = ", ".join(subject) if isinstance(subject, list) else subject
    
    chain = create_question_generator_chain(rag_context=rag_context)
    
    response_text = chain.invoke({
        "n": n,
        "class_name": class_name,
        "subject_str": subject_str,
        "language": language
    })
    return response_text

def evaluate_quiz_format(raw_text: str) -> bool:
    """
    Evaluates if the raw generated quiz text adheres to the expected format.
    Checks for:
    - Presence of "Question:"
    - Presence of A, B, C, D options
    - Presence of "Answer: X"
    """
    if not re.search(r"\d+\.\s*Question:", raw_text):
        print("Evaluation Failed: No question marker found.")
        return False
    
    pattern = r"Question:.*?A\..*?B\..*?C\..*?D\..*?Answer:\s*[ABCD]"
    if not re.search(pattern, raw_text, re.DOTALL):
        print("Evaluation Failed: Incomplete question/options/answer format.")
        return False
    
    print("Evaluation Succeeded: Format appears valid.")
    return True

def parse_questions(raw_text):
    """
    Parses the raw text output from the LLM into a structured list of questions.
    Each question includes the question text, options, and the correct answer.
    """
    questions = []
    pattern = r"\d+\.\s*Question:(.*?)\s*A\.(.*?)\s*B\.(.*?)\s*C\.(.*?)\s*D\.(.*?)\s*Answer:\s*([ABCD])"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    
    for match in matches:
        q = {
            "question": match[0].strip(),
            "options": {
                "A": match[1].strip(),
                "B": match[2].strip(),
                "C": match[3].strip(),
                "D": match[4].strip(),
            },
            "correct": match[5].strip()
        }
        questions.append(q)
    return questions
