

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# -------             Importing Libraries and Modules and function from other pages                --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------



import streamlit as st
import re
import os
from dotenv import load_dotenv # Needed here for initial API key check
from typing import Any

# Import functions from the backend agent module
from backend.student_data import DataStore
from backend.rag_vector_store import initialize_rag_db, get_rag_context
from backend.langgraph_workflow import run_quiz_generation_agent
from backend.quiz_evaluation_graph import run_quiz_evaluation_agent


load_dotenv() # Load .env at the very top of the Streamlit app


# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Quiz Generator", layout="centered")


MONGODB_URI = os.getenv("MONGODB_URI")
data_store = DataStore(mongo_uri=MONGODB_URI)


#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# ------------------             Initialize session state variables                --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------



# Sample RAG Content (for demonstration)
# In a real app, this would come from files, a database, etc.
RAG_SAMPLE_TEXT = """
Class 1-5 Basic Math: Addition is the process of combining two or more numbers to get a total. Subtraction is taking one number away from another. Multiplication is repeated addition, and division is splitting into equal parts. Fractions represent parts of a whole.

Class 6-8 Science: Photosynthesis is the process by which plants make their food using sunlight, water, and carbon dioxide. The human circulatory system includes the heart and blood vessels that transport oxygen and nutrients.

Class 9-12 Physics: Newton's Laws of Motion describe how forces affect an object's movement. The first law states an object at rest stays at rest unless acted upon by a force.

Bachelors: Advanced mathematics includes subjects like linear algebra, calculus, and differential equations. Computer science covers algorithms, data structures, programming paradigms, and software development principles.

Masters: Topics include advanced algorithms, machine learning, artificial intelligence, distributed computing, and data science.
"""



#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# -----------             Initialize FAISS Vector Store using st.cache_resource                --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------


@st.cache_resource
def get_faiss_vector_store():
    """
    Initializes and caches the FAISS vector store.
    This function runs only once per app session.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google Gemini API Key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop() # Stop the app if API key is missing
    
    try:
        vector_store = initialize_rag_db(RAG_SAMPLE_TEXT, api_key)
        print("✅ RAG knowledge base initialized successfully!")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error initializing RAG knowledge base: {e}")
        st.stop() # Stop if RAG initialization fails

# Initialize the vector store at app startup
vector_store = get_faiss_vector_store()




#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# ------------------             Initialize session state variables                --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

def initialize_session_state():
    if "general_topics" not in st.session_state:
        st.session_state["general_topics"] = []
    if "topic_input" not in st.session_state:
        st.session_state["topic_input"] = ""
    if "quiz_started" not in st.session_state:
        st.session_state["quiz_started"] = False
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "answers" not in st.session_state:
        st.session_state["answers"] = []
    if "current_q" not in st.session_state:
        st.session_state["current_q"] = 0
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "preferred_language" not in st.session_state:
        st.session_state.preferred_language = None
    if "student_id" not in st.session_state:
        st.session_state.student_id = None
    if "current_selected_option" not in st.session_state:
        st.session_state["current_selected_option"] = None
    if  "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "expanded_subject" not in st.session_state:
        st.session_state["expanded_subject"] = None
    if "final_update" not in st.session_state:
        st.session_state["final_update"] = 0
    if "personalized_feedback" not in st.session_state:
        st.session_state["personalized_feedback"] = None    

initialize_session_state()

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# ------------------             Initialize class and Title                --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------


# Class to subjects mapping (includes Classes 3–12 with regional languages)

class_subject_map = {
    "Class 3": [ "Math", "EVS", "English", "Hindi", "Marathi", "Tamil", "Telugu", "Punjabi"],
    "Class 4": [ "Math", "EVS", "English", "Hindi", "Bengali", "Kannada", "Urdu", "Gujarati"],
    "Class 5": [ "Math", "EVS", "English", "Hindi", "Sanskrit", "Tamil", "Telugu", "Odia"],
    "Class 6": [ "Math", "Science", "English", "Social Science", "Hindi", "Sanskrit", "Urdu", "Tamil", "Marathi"],
    "Class 7": [ "Math", "Science", "English", "Social Science", "Hindi", "Sanskrit", "Bengali", "Telugu"],
    "Class 8": [ "Math", "Science", "English", "Social Science", "Hindi", "Punjabi", "Kannada", "Malayalam"],
    "Class 9": [ "Math", "Science", "English", "Social Science", "Hindi", "Sanskrit", "Computer Applications", "Tamil"],
    "Class 10": [ "Math", "Science", "English", "Social Science", "Hindi", "Sanskrit", "IT", "Marathi", "Bengali"],
    "Class 11 (Science)": [ "Physics", "Chemistry", "Math", "Biology", "English", "Computer Science", "PE", "Environmental Science"],
    "Class 11 (Commerce)": [ "Accountancy", "Business Studies", "Economics", "Math", "English", "IT", "Entrepreneurship"],
    "Class 11 (Arts)": [ "History", "Geography", "Political Science", "Economics", "Sociology", "Psychology", "English", "Hindi"],
    "Class 12 (Science)": [ "Physics", "Chemistry", "Math", "Biology", "English", "Computer Science", "PE"],
    "Class 12 (Commerce)": [ "Accountancy", "Business Studies", "Economics", "Math", "English", "Entrepreneurship"],
    "Class 12 (Arts)": [ "History", "Geography", "Political Science", "Sociology", "Economics", "Psychology", "Hindi", "English"]
}





#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# ---------------------                         Sidebar Content                    ---------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------



if st.session_state.preferred_language or st.session_state.student_id :
    with st.sidebar:
        st.markdown(f""" <div style='font-size:28px; font-weight:800; color:#945; margin-bottom:15px; font-family: "Arial", sans-serif;'>
            User Profile </div> """, unsafe_allow_html=True )
        
        # This conditional block runs only after a language has been selected.
        if st.session_state.stage >= 1:
            st.markdown(
                f"""<div style='font-size:16px; font-weight:400; color:#589; margin-bottom:15px;'>
                "Preferred Language: 
                </div> """, unsafe_allow_html=True)
            st.markdown(
                f""" <div style='font-size:16px; font-weight:400; color:#1BC2A0; margin-bottom:15px;'>
                {st.session_state.preferred_language} 
                </div> """, unsafe_allow_html=True )
            

            if st.button("Change Language"):
                # Reset the session state variables.
                st.session_state.preferred_language = None
                st.session_state.stage = 0
                st.rerun()

        # This conditional block runs only after a student ID has been entered.
        if st.session_state.stage >= 2 or st.session_state.student_id:
            st.markdown(
                f""" <div style='font-size:16px; font-weight:400; color:#589; margin-bottom:15px;'>
                Student ID: 
                </div> """, unsafe_allow_html=True )
            st.markdown(
                f""" <div style='font-size:16px; font-weight:400; color:#1BC2A0; margin-bottom:15px;'>
                {st.session_state.student_id}
                </div> """, unsafe_allow_html=True )
            


        if st.session_state.student_id:
            student_perf = data_store.get_student_performance(st.session_state.student_id)
            if student_perf:

                st.markdown(
                    f""" <div style='font-size:16px; font-weight:400; color:#589; margin-bottom:15px;'>
                    Class Selected: 
                    </div> """, unsafe_allow_html=True )
                st.markdown(
                    f""" <div style='font-size:16px; font-weight:400; color:#1BC2A0; margin-bottom:15px;'>
                    {student_perf['class']}
                    </div> """, unsafe_allow_html=True )
            
                st.markdown(
                    f"""<div style='font-size:16px; font-weight:800; color:#589; margin-bottom:15px;'>
                    <span style='font-size:20px;font-weight:600'>
                    🏅 Your Past Performance Summary 
                    </span> </div> """, unsafe_allow_html=True)


                # set progreass bar colour green
                st.markdown("""<style> .stProgress > div > div > div > div {
                            background-color: green; } </style>""", unsafe_allow_html=True,)

                # Aggregate overall performance for quick summary
                total = student_perf.get("total_questions_attempted", 0)
                correct = student_perf.get("total_correct_answers", 0)
                overall_pct = (correct / total * 100) if total else 0

                # Big progress bar & percentage
                st.markdown(
                    f"""<div style='font-size:16px; font-weight:400; color:#1BC2A0; margin-bottom:15px;'> 
                    Overall Accuracy: <span style='font-weight:700'>
                    {overall_pct: .2f}%</span> </div>""", unsafe_allow_html=True)
                
                prog_val = float(overall_pct) / 100 if overall_pct else 0.0
                st.progress(prog_val)

                st.markdown(f"""<div style='font-size:16px; font-weight:800; color:#589; margin-bottom:15px;'>
                            <span style='font-size:20px;font-weight:600'> 
                            📚 Subject Performance </span></div>""", unsafe_allow_html= True)
                
                # Prepare subject stats for gap analysis
                subject_rows = []
                for subject, subj_stats in student_perf.get("subjects", {}).items():
                    subj_total = subj_stats.get("total_attempts", 0)
                    subj_corr = subj_stats.get("correct_count", 0)
                    subj_pct = int(subj_corr / subj_total * 100) if subj_total else 0
                    subject_rows.append((subject, subj_pct, subj_corr, subj_total, subj_stats.get("topics", {})))

                
                subject_pct_list = [(subject, subj_pct) for subject, subj_pct, _, _, _ in subject_rows]
                lowest_subjects = sorted(subject_pct_list, key=lambda x: x[1])[:2]


                def toggle_subject(subject_name):
                    if st.session_state.get("expanded_subject") == subject_name:
                        st.session_state["expanded_subject"] = None
                    else:
                        st.session_state["expanded_subject"] = subject_name

                # Display subject bars and expanders
                for subject, subj_pct, subj_corr, subj_total, topics in subject_rows:
                    cols = st.columns([5, 5, 3])
                    cols[0].markdown(f"""<div style='font-size:16px; font-weight:700; color:#589; margin-bottom:15px;'><span style='font-size:18px;font-weight:600'> {subject} </span></div>""", unsafe_allow_html= True)
                    cols[1].progress(subj_pct / 100 if subj_pct else 0.0)
                    cols[2].markdown(f"""<div style='font-size:16px; font-weight:700; color:#1BC2A0; margin-bottom:15px;'>
                                     <span style='font-size:16px;font-weight:600'>{subj_pct}%</span> </div>""", unsafe_allow_html=True)
                    

                    # Toggle button
                    expanded = st.session_state.get("expanded_subject")


                    button_label = "Hide Topics" if expanded == subject else "Show Topics"
                    if cols[0].button(button_label, key=f"toggle_topics_{subject}", on_click=toggle_subject, args=(subject,)):
                        pass


                    # Expand/collapse per subject
                    if st.session_state["expanded_subject"] == subject:
                        with st.expander(f"Topic Performance for {subject}", expanded=True):
                            # Topic bars, one line each
                            for topic, t_stats in topics.items():
                                topic_total = t_stats.get("total_attempts", 0)
                                topic_corr = t_stats.get("correct_count", 0)
                                topic_pct = int(topic_corr / topic_total * 100) if topic_total else 0
                                tcols = st.columns([5, 5, 3])
                                tcols[0].markdown(f"{topic}")
                                tcols[1].progress(topic_pct / 100 if topic_pct else 0.0)
                                tcols[2].markdown(f"{topic_pct}%", unsafe_allow_html=True)

                    st.markdown("")  # spacing

                gap_names = [subname for subname, _ in lowest_subjects]
                if gap_names:
                    st.markdown("---")
                    st.markdown(f"""<div style='font-size:16px; font-weight:700; color:#589; margin-bottom:15px;'>
                                <span style='font-size:16px;font-weight:600'>
                                Subject Gap: (Two least perform Subject)  </span> </div>""", unsafe_allow_html= True)
                    st.markdown(
                        " &mdash; ".join([
                        f"""<span style='font-size:16px; font-weight:700; color:#1BC2A0; margin-bottom:15px;'>
                        {name} </span>""" for name in gap_names
                        ]), unsafe_allow_html=True )

                    st.markdown("---")



        
        # A single button to reset the entire application flow.
        if st.button("Reset All Settings"):
            st.session_state.clear()
            st.rerun()



#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# -----------------            STEP 1: Region / Language Selection                  ---------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------



language_options = ["-- Select Language --", "English", "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati",
                    "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese", "Maithili", "Kashmiri", "Nepali", 
                    "Meitei", "Bodo", "Dogri", "Santali", "Konkani", "Sindhi", "Sanskrit", "Mandarin Chinese", "Spanish", 
                    "French", "German", "Russian", "Portuguese", "Japanese", "Arabic", "Italian", "Turkish", "Korean", 
                    "Vietnamese", "Thai", "Indonesian", "Dutch", "Swahili",  "Ukrainian", "Persian", "Filipino", "Hausa", 
                    "Bulgarian", "Croatian", "Czech", "Danish", "Finnish", "Greek", "Hebrew", "Hungarian", "Indonesian", 
                    "Latvian", "Norwegian", "Polish", "Romanian", "Swedish"]


# --- TITLE ---
st.title("📚 AI-Powered Quiz for Rural Learners")

if st.session_state.stage == 0:

    st.markdown("### 🌐 Select Your Preferred Language for the Website")
    selected_language = st.selectbox( "Choose Language", options=language_options, key="language_selector_widget")

    if selected_language == "-- Select Language --":
        st.warning("⚠️ Please select a language to continue.")
    else:
        st.session_state.preferred_language = selected_language
        st.session_state.stage = 1  # Move to the next stage
        st.rerun()
    





#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# -----------------                      STEP 2: Student ID Input                     --------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

elif st.session_state.stage == 1 :
    if st.session_state.student_id:
        st.session_state.stage = 2
        st.rerun()
    else:
        st.markdown("### Enter Your Student ID")
        student_id = st.text_input("Please enter your Student ID:", key="student_id_input")

        def is_valid_student_id(sid):
            return bool(re.fullmatch(r'[A-Za-z0-9]+', sid))

        if student_id:
            if is_valid_student_id(student_id):
                st.session_state.student_id = student_id
                st.session_state.stage = 2 # Move to the final stage
                st.success(f"✅ Student ID {student_id} accepted. Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid Student ID. Use only letters and numbers (no spaces or special characters).")
        else:
            st.info("ℹ️ Please enter your Student ID to continue.")







#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# --------------------                   STEP 3: Quiz Input                     --------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------


elif st.session_state.stage == 2 :

    if student_perf is None:

        # New Student : Full Class Selection as Before
        st.markdown(
            f""" <div style='font-size:28px; font-weight:800; color:#036959; margin-bottom:5px; font-family: "Arial", sans-serif;'> 
                🎓 Select Your Class ( This will be your class for this ID permanently)
                </div> """, unsafe_allow_html=True )
        class_selected = st.selectbox("", options=["-- Select Class --"]  + list(class_subject_map.keys()))
        if class_selected == "-- Select Class --":
            st.warning("⚠️ Please select a valid class to continue.")
            st.stop()
        st.session_state["class_selected"] = class_selected

    else:
        # Existing Student 
        stored_class = student_perf.get("class", None)
        st.session_state["class_selected"] = class_selected = stored_class
        st.success(f"Your registered class: **{stored_class}** (locked)")
        

    choice = st.radio("Choose quiz mode:", [ "Choose Subject", "General (Choose topic of your choice)"])

    if choice == "Choose Subject":
        auto_detect = st.checkbox("📊 Auto-detect subject based on my past performance")

        # Subject selection (only shown if auto-detect is NOT checked)
        if not auto_detect :
            subjects_for_class = class_subject_map.get(class_selected, [])
            st.markdown(f""" <div style='font-size:28px; font-weight:800; color:#036959; margin-bottom:5px; font-family: "Arial", sans-serif;'> 
                📘 Select Subject </div> """, unsafe_allow_html=True )
            selected_subject = st.selectbox("", options=["-- Select Subject --"] + subjects_for_class)
        else:
            selected_subject = None  # Explicitly mark as unused

    elif choice == "General (Choose topic of your choice)":
        auto_detect = False


        def add_topic():
            cleaned = st.session_state.topic_input.strip()
            if cleaned:
                if cleaned not in st.session_state["general_topics"]:
                    st.session_state["general_topics"].append(cleaned)
                else:
                    st.info("ℹ️ Topic already added.")
            else:
                st.warning("⚠️ Topic cannot be empty.")
            st.session_state["topic_input"] = ""  # This works inside callback

        st.markdown(f""" <div style='font-size:20px; font-weight:800; color:#036959; margin-bottom:5px; font-family: "Arial", sans-serif;'> 
                ✏️ Enter a topic and press Enter </div> """, unsafe_allow_html=True )
        st.text_input("", key="topic_input", on_change=add_topic )


        col1, col2, col3, col4, col5 = st.columns(5)


        with col5:
            if st.button("Clear Topics"):
                st.session_state["general_topics"] = []

        with col1: # Show all added topics
            if st.session_state["general_topics"]:
                st.markdown("Topics Selected:")
                for idx, topic in enumerate(st.session_state["general_topics"], 1):
                    st.write(f"{idx}. {topic}")
            else:
                st.info("No topics added yet.")

            st.session_state["selected_subject"] = st.session_state["general_topics"]
    #--------------------------------------------------------------------------------------------------------------------------------------
    # ---------------                Step 5: Auto Detect or Subject Selection               --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------

    


    # --- NUMBER OF QUESTIONS ---
    num_questions = st.slider("❓ How many questions do you want?", min_value=5, max_value=30, step=1, value=5)



    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    # --------------------                   SUBMIT Button                     --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------



    col1, col2, col3, col4 = st.columns([1, 1, 2, 1 ])
    with col3:
        if st.button("Start Quiz 🚀"):

            # Determine the subject to use
            if choice == "General (Choose topic of your choice)":
                selected_subject = st.session_state.get("general_topics", [])

                # Validation: must have topics and a valid level
                if not selected_subject:
                    st.warning("⚠️ Please add at least one topic.")
                    st.stop()


            elif not auto_detect:
                selected_subject = selected_subject  # From earlier selectbox
                class_selected = class_selected
            else:

                # AUTO-DETECT MODE: Get weakest subject and topics from DB
                student_perf = data_store.get_student_performance(st.session_state.student_id)
                if not student_perf or "subjects" not in student_perf:
                    st.error("❌ No past performance data found. Please take a quiz first before using auto-detect.")
                    st.stop()

                # 1. Find min accuracy across subjects
                subject_accs = []
                for subj, stats in student_perf["subjects"].items():
                    total = stats.get("total_attempts", 0)
                    if total > 0:
                        acc = stats.get("correct_count", 0) / total
                        subject_accs.append((subj, acc))
                if not subject_accs:
                    st.error("❌ No subjects with attempts found.")
                    st.stop()

                min_acc = min(acc for _, acc in subject_accs)
                lowest_subjects = [subj for subj, acc in subject_accs if acc == min_acc]

                # 2. For each lowest subject, pick up to 2 weakest topics <90% accuracy
                selected_subject = []
                for subj in lowest_subjects:
                    topics_data = student_perf["subjects"][subj].get("topics", {})
                    topic_accs = [
                        (t, tdata.get("correct_count", 0) / max(tdata.get("total_attempts", 1), 1))
                        for t, tdata in topics_data.items()
                        if tdata.get("total_attempts", 0) > 0
                    ]
                    weak_topics = [t for t, acc in sorted(topic_accs, key=lambda x: x[1]) if acc < 0.9]
                    if weak_topics:
                        for topic in weak_topics[:3]:
                            selected_subject.append(f"{subj} - {topic}")
                    else:
                        selected_subject.append(subj)

                
                


            # Validation
            if not auto_detect:
                if isinstance(selected_subject, list):
                    if not selected_subject:
                        st.warning("⚠️ Please add at least one topic.")
                        st.stop()
                elif not selected_subject or selected_subject.strip() == "":
                    st.warning("⚠️ Please choose a subject or enable auto-detect based on past performance.")
                    st.stop()

            # Clear previous quiz state
            for key in ["questions", "answers", "current_q"]:
                st.session_state.pop(key, None)

            # Store session state
            st.session_state['quiz_started'] = True
            st.session_state['num_questions'] = num_questions
            st.session_state['class'] = class_selected
            st.session_state['subject'] = selected_subject
            st.session_state['auto_detect'] = auto_detect
            st.session_state.stage = 3

            st.success("✅ Quiz initialized! Proceeding to question generation...")
            st.rerun()





#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
# --------------------                   Start QUIZ after Click                     --------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------


elif st.session_state.stage == 3:
    
    if st.session_state.get('quiz_started') and not st.session_state.get("questions"):
        with st.spinner("🎯 Generating questions..."):
            subject_to_generate = st.session_state["subject"]
            
            # Pass the auto_detect flag as the include_rag parameter AND the vector_store
            parsed = run_quiz_generation_agent(
                n=st.session_state["num_questions"],
                class_name=st.session_state["class"],
                subject=subject_to_generate,
                language=st.session_state['preferred_language'],
                include_rag=st.session_state['auto_detect'], # Pass the flag here!
                vector_store=vector_store # Pass the initialized FAISS vector store
            )

            if parsed:
                st.session_state["questions"] = parsed
                st.session_state["current_q"] = 0
                st.session_state["answers"] = [""] * len(parsed)  # Initialize answers list with empty strings
                st.session_state["current_selected_option"] = None
                st.success("✅ Questions Ready!")
                st.rerun()
            else:
                st.error("❌ Failed to generate questions. This might be due to an LLM generation issue or parsing problem. Please try again or adjust your input.")
                st.session_state['quiz_started'] = False




    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    # ---------------                   DISPLAY ONE QUESTION AT A TIME                     --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------

    if st.session_state.get("questions") and st.session_state["current_q"] < len(st.session_state["questions"]):
        qn = st.session_state["questions"][st.session_state["current_q"]]
        
        st.markdown(
            f"""
            <div style='font-size:28px; font-weight:800; color:#645; margin-bottom:15px;'>
                Question {st.session_state['current_q'] + 1}: {qn['question']}
            </div>
            """, 
            unsafe_allow_html=True
        )

        
        options = qn["options"]
        
        # Pre-select the user's previous answer if they revisit the question
        # This also helps track the current selection.
        current_answer = st.session_state["answers"][st.session_state["current_q"]]
        
        # Create the radio button group for options
        selected_option = st.radio(
            "Select your answer:",
            options.keys(),
            format_func=lambda x: f"{x}. {options[x]}",
            index=list(options.keys()).index(current_answer) if current_answer in options else None
        )

        # Update the current_selected_option in session state
        st.session_state.current_selected_option = selected_option
        
        st.divider()

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⬅️ Previous", disabled=(st.session_state["current_q"] == 0)):
                # Save the current answer before moving back
                if st.session_state.current_selected_option:
                    st.session_state["answers"][st.session_state["current_q"]] = st.session_state.current_selected_option
                st.session_state["current_q"] -= 1
                st.rerun()

        with col3:
            if st.session_state["current_q"] < len(st.session_state["questions"]) - 1:
                if st.button("Next ➡️"):
                    # Save the current answer before moving next
                    if st.session_state.current_selected_option:
                        st.session_state["answers"][st.session_state["current_q"]] = st.session_state.current_selected_option
                    st.session_state["current_q"] += 1
                    st.rerun()
            else:
                if st.button("Submit Quiz ✅"):
                    # Save the final answer and transition to the results page
                    if st.session_state.current_selected_option:
                        st.session_state["answers"][st.session_state["current_q"]] = st.session_state.current_selected_option
                    # Set flag to indicate quiz submission
                    st.session_state["quiz_submitted"] = True 
                    st.session_state["current_q"] += 1 # A dummy value to trigger the next block
                    st.session_state.stage = 4
                    st.rerun()




    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    # --------------------                   SHOW FINAL SCORE                     --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------


elif st.session_state.stage == 4:
    #--------------------------------------------------------------------------------------------------------------------------------------
    # --------------------                 Display Performance                     --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    

    def display_performance_report(evaluation_results, questions):
        total = len(evaluation_results)
        correct = sum(r["is_correct"] for r in evaluation_results)
        incorrect = total - correct
        accuracy = (correct / total) * 100 if total > 0 else 0

        # Aggregate performance by subject and topic
        subj_stats = {}
        topic_stats = {}

        for r in evaluation_results:
            subj = r["subject"]
            topic = r["topic"]

            subj_stats.setdefault(subj, {"total": 0, "correct": 0})
            topic_stats.setdefault(topic, {"total": 0, "correct": 0})

            subj_stats[subj]["total"] += 1
            topic_stats[topic]["total"] += 1

            if r["is_correct"]:
                subj_stats[subj]["correct"] += 1
                topic_stats[topic]["correct"] += 1

        # Header
        st.markdown("## 📊 Quiz Performance Report")
        st.markdown(f"### Here is a summary of your recent quiz performance:")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Questions", total)
        col2.metric("Correct Answers", correct)
        col3.metric("Incorrect Answers", incorrect)
        st.markdown(f"**Overall Accuracy:** {accuracy:.1f}%")

        st.markdown("---")

        # Performance by subject
        st.markdown("### 🎯 Performance Breakdown by Subject:")
        for subj, stats in subj_stats.items():
            subj_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            st.markdown(f"* **{subj}**: {stats['correct']} out of {stats['total']} correct ({subj_acc:.0f}% accuracy)")

        st.markdown("---")

        # Performance by topic
        st.markdown("### 📚 Performance Breakdown by Topic:")
        for topic, stats in topic_stats.items():
            topic_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            st.markdown(f"* **{topic}**: {stats['correct']} out of {stats['total']} correct ({topic_acc:.0f}% accuracy)")

        st.markdown("---")

        

        # Detailed Question Results
        st.markdown("### 📝 Detailed Question Results")
        for idx, res in enumerate(evaluation_results):
            status_icon = "✅" if res["is_correct"] else "❌"
            color = "green" if res["is_correct"] else "red"

            selected_letter = res.get("user_answer", "N/A")
            correct_letter = res.get("correct_answer", "N/A")
            option_dict = questions[idx]["options"]

            selected_text = option_dict.get(selected_letter, "Not answered")
            correct_text = option_dict.get(correct_letter, "Unavailable")

            st.markdown(f"**Q{idx + 1}:** {res['question']}")
            st.markdown(
                f"* Your Answer: `{selected_letter}`. {selected_text} | "
                f"Correct Answer: `{correct_letter}`. {correct_text} | "
                f"<span style='color:{color}'>{status_icon} {'Correct' if res['is_correct'] else 'Incorrect'}</span>",
                unsafe_allow_html=True
            )
            st.markdown(f"* Subject: **{res['subject']}**, Topic: **{res['topic']}**")
            st.markdown("---")

        
        st.markdown("### 💡 Personalized Feedback:")
        st.write(st.session_state["personalized_feedback"])
        st.markdown("---")




    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    # --------------------             Evaluate Results and save performance                   --------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------


    if st.session_state["quiz_submitted"] == True:
        # Run evaluation and annotation only once
        if "evaluation_results" not in st.session_state or "performance_report" not in st.session_state:
            
            with st.spinner("Evaluating quiz, generating performance report, and updating database..."):
                final_state = run_quiz_evaluation_agent(
                    student_id=st.session_state.get("student_id"),
                    questions=st.session_state["questions"],
                    answers=st.session_state["answers"],
                    language=st.session_state.get("preferred_language", "English"),
                    data_store=data_store,
                    class_selected=st.session_state.get("class"),
                    selected_subject=st.session_state.get("subject"),
                    general_topics=st.session_state.get("general_topics", []),
                    auto_detect=st.session_state.get("auto_detect", False)
                )

                # Save results to session state
                st.session_state["evaluation_results"] = final_state["evaluation_results"]
                st.session_state["performance_report"] = final_state["performance_report"]
                st.session_state["personalized_feedback"] = final_state["feedback"]

                st.success("✅ Evaluation completed and saved!")

        # Display results with existing UI
        if "evaluation_results" in st.session_state:
            display_performance_report(st.session_state["evaluation_results"], st.session_state["questions"])
            if st.session_state["final_update"] == 0:
                st.session_state["final_update"] = 1
                st.rerun()
        else:
            st.info("No evaluation results available yet.")

        st.session_state["quiz_submitted"] = False

    if st.button("Restart Quiz"):
        st.session_state.clear()
        st.rerun()
