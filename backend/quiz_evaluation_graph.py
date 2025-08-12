# backend/quiz_evaluation_graph.py

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

# Import existing backend logic
from backend.performance_evaluator import (
    evaluate_answers,
    generate_performance_report,
    generate_personalized_feedback,
    extract_or_generate_subject_topic
)
from backend.student_data import DataStore


# -------------------- STATE FOR EVALUATION GRAPH --------------------
class EvaluationState(TypedDict):
    student_id: str
    questions: List[dict]          # Parsed quiz questions
    answers: List[str]             # User's answers (A/B/C/D)
    language: str                  # Preferred language for reports
    evaluation_results: List[dict] # Detailed grading results per question
    performance_report: str        # LLM-generated performance report
    feedback: str                   # LLM-generated personalized feedback
    data_store: Any                 # DataStore instance
    class_selected: str             # Class name
    selected_subject: str           # Subject name OR list
    general_topics: List[str]       # For topic classification if general class
    auto_detect : bool              # checking if auto detect selected or not


# ======================= NODES =======================

def add_subject_topic_node(state: EvaluationState) -> EvaluationState:
    """
    Enrich each question with subject & topic.
    - Auto-detect mode: we already have subject/topic pairs, assign directly (no LLM).
    - Other modes: detect using LLM.
    """
    print("--- Evaluation Graph: Adding subject/topic to questions ---")
    enriched_questions = []

    for i, q in enumerate(state["questions"]):

        # Auto-detect mode: selected_subject is a list of "Subject - Topic" strings
        if state.get("auto_detect", False):
            try:
                subj, topic = state["selected_subject"][i].split(" - ", 1)
                q["subject"] = subj.strip()
                q["topic"] = topic.strip()
            except Exception:
                # Fallback in case of mismatch
                q["subject"] = "Unknown"
                q["topic"] = "Unknown"

        else:
            # Manual or general mode → use LLM extraction
            subj, topic = extract_or_generate_subject_topic(
                q["question"],
                state["class_selected"],
                state["selected_subject"],
                state["general_topics"]
            )
            q["subject"] = subj
            q["topic"] = topic

        enriched_questions.append(q)

    state["questions"] = enriched_questions
    return state



def evaluate_answers_node(state: EvaluationState) -> EvaluationState:
    """
    Grades the user's answers against the correct answers.
    """
    print("--- Evaluation Graph: Evaluating answers ---")
    evaluation_results = evaluate_answers(state["questions"], state["answers"])
    state["evaluation_results"] = evaluation_results
    return state


def generate_report_node(state: EvaluationState) -> EvaluationState:
    """
    Generates a performance report using LLM.
    """
    print("--- Evaluation Graph: Generating performance report ---")
    report = generate_performance_report(state["evaluation_results"], language=state["language"])
    state["performance_report"] = report
    return state


def generate_feedback_node(state: EvaluationState) -> EvaluationState:
    """
    Generates personalized feedback using LLM.
    """
    print("--- Evaluation Graph: Generating personalized feedback ---")
    feedback = generate_personalized_feedback(state["evaluation_results"], language=state["language"])
    state["feedback"] = feedback
    return state


def update_db_node(state: EvaluationState) -> EvaluationState:
    """
    Updates the student's performance record in MongoDB.
    """
    print("--- Evaluation Graph: Updating database with results ---")
    datastore: DataStore = state["data_store"]
    datastore.update_student_performance( student_id= state["student_id"],
                                        class_name=state["class_selected"], 
                                        evaluation_results=state["evaluation_results"])
    return state


# ======================= BUILD GRAPH =======================

def build_quiz_evaluation_graph():
    """
    Builds and compiles the LangGraph workflow for quiz evaluation.
    """
    workflow = StateGraph(EvaluationState)

    # Add nodes
    workflow.add_node("add_subject_topic", add_subject_topic_node)
    workflow.add_node("evaluate_answers", evaluate_answers_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("generate_feedback", generate_feedback_node)
    workflow.add_node("update_db", update_db_node)

    # Set execution flow
    workflow.set_entry_point("add_subject_topic")
    workflow.add_edge("add_subject_topic", "evaluate_answers")
    workflow.add_edge("evaluate_answers", "generate_report")
    workflow.add_edge("generate_report", "generate_feedback")
    workflow.add_edge("generate_feedback", "update_db")
    workflow.add_edge("update_db", END)

    return workflow.compile()


# ======================= RUN GRAPH FUNCTION =======================

def run_quiz_evaluation_agent(
        student_id: str,
        questions: List[dict],
        answers: List[str],
        language: str,
        data_store: DataStore,
        class_selected: str,
        selected_subject: str,
        general_topics: List[str],
        auto_detect: bool = False
) -> EvaluationState:
    """
    Runs the evaluation workflow and returns the final state.
    """
    app = build_quiz_evaluation_graph()
    initial_state: EvaluationState = {
        "student_id": student_id,
        "questions": questions,
        "answers": answers,
        "language": language,
        "evaluation_results": [],
        "performance_report": "",
        "feedback": "",
        "data_store": data_store,
        "class_selected": class_selected,
        "selected_subject": selected_subject,
        "general_topics": general_topics,
        "auto_detect": auto_detect,
    }
    final_state = app.invoke(initial_state)
    return final_state
