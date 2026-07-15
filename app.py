"""
AI Powered Intelligent Hiring Assistant
Developed by Devesh Sain

This app combines the work done in our notebooks:
1. Resume Classification (SVM + TF-IDF)
2. Resume - JD Similarity (Sentence Transformers)
3. Skill Gap Analysis (hardcoded skills database)
4. Final Hiring Score
5. Rule Based Feedback
6. AI Feedback (OpenRouter)
7. Candidate Chatbot (OpenRouter + Knowledge Base)
"""

# -------------------------------------------------
# Import Libraries
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np          # NEW: needed for similarity sorting in the RAG pipeline
import joblib
import pdfplumber
import re
import io   
import streamlit as st



# NEW: needed to build the PDF report in memory

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# NEW: reportlab is used to create the downloadable PDF report
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape   # NEW: used to safely insert AI text into the PDF

# -------------------------------------------------

# -------------------------------------------------


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="AI Powered Intelligent Hiring Assistant", page_icon="🎯", layout="wide")

# -------------------------------------------------
# Custom CSS (keeps things simple, just colors, spacing and rounding)
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1c29 0%, #12141f 100%);
}
[data-testid="stSidebar"] * {
    color: #f5f5f7;
}
[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #3a3d52;
    background-color: transparent;
    color: #e8e8ec;
    text-align: left;
    padding: 0.55rem 1rem;
    margin-bottom: 0.4rem;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton>button:hover {
    border-color: #6C63FF;
    color: #6C63FF;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"] {
    background-color: #6C63FF;
    border-color: #6C63FF;
    color: white;
}

/* Main action buttons */
.stButton>button[kind="primary"] {
    background-color: #6C63FF;
    border-color: #6C63FF;
    border-radius: 8px;
}
.stButton>button[kind="primary"]:hover {
    background-color: #574fd6;
    border-color: #574fd6;
}

/* Progress bar accent */
.stProgress > div > div > div > div {
    background-color: #6C63FF;
}

/* Rounder expanders and containers */
[data-testid="stExpander"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Hardcoded Skills Database
# (Same list used in the notebook, skills.csv is NOT used)
# -------------------------------------------------
skills_database = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "R", "Go", "Rust", "PHP", "Kotlin", "Swift",
    # Web Development
    "HTML", "CSS", "Bootstrap", "Tailwind CSS", "React", "Angular", "Vue", "Next.js", "Node.js", "Express",
    "Django", "Flask", "FastAPI", "Spring", "Hibernate", ".NET", "ASP.NET",
    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "NoSQL",
    # Data Science & ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "CV", "Scikit-Learn", "TensorFlow",
    "Keras", "PyTorch", "XGBoost", "LightGBM",
    # Python Libraries
    "NumPy", "Pandas", "Matplotlib", "Seaborn", "SciPy", "OpenCV",
    # Big Data
    "Spark", "PySpark", "Hadoop", "Hive", "Kafka",
    # Cloud
    "AWS", "Azure", "GCP", "EC2", "S3", "Lambda",
    # DevOps
    "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab", "Linux", "Bash",
    # GenAI
    "OpenAI", "LLM", "RAG", "LangChain", "LlamaIndex", "Transformers", "Hugging Face",
    "FAISS", "ChromaDB", "Prompt Engineering",
    # Visualization
    "Power BI", "Tableau", "Excel", "Looker",
    # Mobile
    "Android", "Flutter", "React Native",
    # Testing
    "JUnit", "Selenium", "PyTest", "Unit Testing",
    # Networking & Security
    "Cyber Security", "Network Security", "Penetration Testing",
    # Software Engineering
    "Agile", "Scrum", "Project Management", "System Design", "REST API", "Microservices",
    # Soft Skills
    "Communication", "Leadership", "Problem Solving", "Critical Thinking",
    "Teamwork", "Time Management", "Presentation Skills", "Analytical Thinking"
]


# -------------------------------------------------
# Load Saved ML Models (Resume Classification)
# -------------------------------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("resume_classifier.pkl")
    tfidf = joblib.load("tfidf.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return classifier, tfidf, encoder


# Load Sentence Transformer Model (for Similarity)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Load Knowledge Base CSV (for Chatbot + AI Feedback)
@st.cache_data
def load_knowledge_base():
    return pd.read_csv("knowledge_base.csv")


# ===================================================
# NEW: RAG (Retrieval-Augmented Generation) PIPELINE
# ===================================================
# The idea of RAG is simple:
#   1. Turn every row of our knowledge base into a small text "document".
#   2. Convert every document into a vector (embedding) ONE TIME and keep it
#      in memory (using Streamlit caching so it is not repeated on every click).
#   3. When the user asks a question, convert the question into a vector too.
#   4. Compare the question vector with all document vectors using cosine
#      similarity and pick the Top-5 most similar documents.
#   5. Give those Top-5 documents to the LLM as "retrieved knowledge" so it
#      can answer using real facts instead of guessing (hallucinating).
# ===================================================

# -------------------------------------------------
# STEP 1: Convert every knowledge_base.csv row into one text document
# Example output for a row:
#   "Skill: Python
#    Description: ...
#    Importance: High
#    Learning Resource: ...
#    Interview Tip: ...
#    Difficulty: Medium"
# -------------------------------------------------
@st.cache_data
def create_kb_documents(knowledge_df):
    documents = []  # this list will hold one text "document" per row

    for _, row in knowledge_df.iterrows():
        doc_text = (
            f"Skill: {row['Skill']}\n"
            f"Description: {row['Description']}\n"
            f"Importance: {row['Importance']}\n"
            f"Learning Resource: {row['Learning Resource']}\n"
            f"Interview Tip: {row['Interview Tip']}\n"
            f"Difficulty: {row['Difficulty']}"
        )
        documents.append(doc_text)

    return documents


# -------------------------------------------------
# STEP 2: Convert all documents into embeddings (numbers that represent meaning)
# st.cache_resource makes sure this heavy step runs ONLY ONCE per app session,
# and the resulting embeddings are kept in memory for every chatbot question.
# The underscore before "_embedding_model" tells Streamlit "do not try to hash
# this object" since a loaded AI model cannot be hashed.
# -------------------------------------------------
@st.cache_resource
def create_kb_embeddings(_embedding_model, documents):
    embeddings = _embedding_model.encode(documents, show_progress_bar=False)
    return embeddings


# -------------------------------------------------
# STEP 3 + 4: Retrieve the Top-K most relevant knowledge base documents
# for a given user question using cosine similarity.
# This is the "Retrieval" part of Retrieval-Augmented Generation.
# -------------------------------------------------
def retrieve_relevant_knowledge(user_question, embedding_model, kb_embeddings, documents, top_k=5):
    # Convert the user's question into an embedding (same vector space as the KB)
    question_embedding = embedding_model.encode([user_question])

    # Compare the question embedding with every knowledge base embedding
    similarity_scores = cosine_similarity(question_embedding, kb_embeddings)[0]

    # Get the indexes of the Top-K highest similarity scores (most relevant first)
    top_indexes = np.argsort(similarity_scores)[::-1][:top_k]

    # Pick out the actual text documents for those top indexes
    retrieved_docs = [documents[i] for i in top_indexes]
    return retrieved_docs


# -------------------------------------------------
# STEP 5: Join the retrieved documents into one context block of text
# that will be inserted into the LLM prompt.
# -------------------------------------------------
def build_rag_context(retrieved_docs):
    if not retrieved_docs:
        return "No relevant knowledge base entries were found."
    return "\n\n".join(retrieved_docs)


# -------------------------------------------------
# Text Extraction from PDF
# -------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------------------------------------
# Clean Resume / JD Text (same as used in Preprocessing notebook)
# -------------------------------------------------
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# -------------------------------------------------
# Predict Resume Category using saved SVM model
# -------------------------------------------------
def predict_category(clean_text, classifier, tfidf, encoder):
    resume_vector = tfidf.transform([clean_text])
    prediction = classifier.predict(resume_vector)
    category = encoder.inverse_transform(prediction)
    return category[0]


# -------------------------------------------------
# Calculate Resume - JD Similarity using Sentence Transformer
# -------------------------------------------------
def get_similarity_score(clean_resume_text, clean_jd_text, embedding_model):
    resume_embedding = embedding_model.encode(clean_resume_text)
    jd_embedding = embedding_model.encode(clean_jd_text)
    similarity = cosine_similarity([resume_embedding], [jd_embedding])
    return float(similarity[0][0] * 100)


# -------------------------------------------------
# Extract Skills from Text using hardcoded skills_database
# -------------------------------------------------
def extract_skills(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+.-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    found_skills = set()
    for skill in skills_database:
        skill_lower = skill.lower()
        skill_pattern = re.sub(r"[^a-z0-9#+.-]", " ", skill_lower)
        if skill_pattern in text:
            found_skills.add(skill)

    return sorted(found_skills)


# -------------------------------------------------
# Compare Resume Skills vs JD Skills
# -------------------------------------------------
def compare_skills(resume_skills, jd_skills):
    matched_skills = sorted(set(resume_skills).intersection(jd_skills))
    missing_skills = sorted(set(jd_skills).difference(resume_skills))
    extra_skills = sorted(set(resume_skills).difference(jd_skills))

    if len(jd_skills) == 0:
        skill_match = 0
    else:
        skill_match = (len(matched_skills) / len(jd_skills)) * 100

    return matched_skills, missing_skills, extra_skills, skill_match


# -------------------------------------------------
# Final Hiring Score = 0.6 x Skill Match + 0.4 x Resume Similarity
# -------------------------------------------------
def calculate_final_score(skill_match, similarity_score):
    return (0.6 * skill_match) + (0.4 * similarity_score)


# -------------------------------------------------
# Rule Based Feedback
# -------------------------------------------------
def rule_based_feedback(final_score, missing_skills):
    if final_score >= 75:
        level = "Excellent Match"
    elif final_score >= 50:
        level = "Good Match"
    else:
        level = "Needs Improvement"

    if len(missing_skills) > 0:
        feedback = f"{level}. The candidate should focus on learning: {', '.join(missing_skills)}"
    else:
        feedback = f"{level}. No major skill gaps found."

    return level, feedback


# -------------------------------------------------
# Retrieve Knowledge Base Info for Missing Skills
# -------------------------------------------------
def get_knowledge_context(missing_skills, knowledge_df):
    context = ""
    for skill in missing_skills:
        rows = knowledge_df[knowledge_df["Skill"].str.lower() == skill.lower()]
        if len(rows) > 0:
            row = rows.iloc[0]
            context += (
                f"\nSkill : {row['Skill']}\nDescription : {row['Description']}\n"
                f"Importance : {row['Importance']}\nLearning Resource : {row['Learning Resource']}\n"
                f"Interview Tip : {row['Interview Tip']}\nDifficulty : {row['Difficulty']}\n"
                "---------------------------------------\n"
            )
    return context


# -------------------------------------------------
# Create OpenRouter Client
# -------------------------------------------------
API_KEY = st.secrets["OPENROUTER_API_KEY"]
def get_openrouter_client(api_key):
    return OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")


# -------------------------------------------------
# Generate AI Feedback using OpenRouter
# -------------------------------------------------
def generate_ai_feedback(client, category, similarity_score, skill_match,
                          matched_skills, missing_skills, knowledge_context):

    prompt = f"""
You are an experienced HR Recruiter and Technical Interviewer.

Analyze the following candidate.

Candidate Category:
{category}

Resume Similarity:
{similarity_score:.2f}%

Skill Match:
{skill_match:.2f}%

Matched Skills:
{', '.join(matched_skills) if matched_skills else 'None'}

Missing Skills:
{', '.join(missing_skills) if missing_skills else 'None'}

Retrieved Knowledge:
{knowledge_context}

Generate a professional report with these headings:
1. Candidate Summary
2. Strengths
3. Weaknesses
4. Explanation of Missing Skills
5. Personalized Learning Roadmap
6. Interview Preparation Tips
7. Hiring Recommendation

Keep the report professional and easy to understand.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=700
    )
    return response.choices[0].message.content


# -------------------------------------------------
# Chatbot Reply using OpenRouter
# UPDATED: now uses BOTH the resume analysis AND the retrieved knowledge
# base context (RAG) instead of only the resume analysis.
# -------------------------------------------------
def get_chatbot_reply(client, user_question, analysis_context, retrieved_context, chat_history):
    # This system prompt tells the LLM exactly how to behave:
    # only answer from the given context, and admit when info is missing
    # instead of making things up (hallucinating).
    system_message = (
        "You are an experienced HR recruiter and career mentor. Answer ONLY using "
        "the provided resume analysis and retrieved knowledge. If the answer is not "
        "available in the retrieved knowledge, clearly say that the information is "
        "unavailable instead of hallucinating.\n\n"
        f"Resume Analysis:\n{analysis_context}\n\n"
        f"Retrieved Knowledge (Top matching entries from the knowledge base):\n{retrieved_context}"
    )

    messages = [{"role": "system", "content": system_message}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=messages,
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].message.content


# -------------------------------------------------
# Build Analysis Context Text (used by chatbot)
# -------------------------------------------------
def build_analysis_context():
    s = st.session_state
    return (
        f"Candidate Category : {s.predicted_category}\n"
        f"Resume Similarity : {s.similarity_score:.2f}%\n"
        f"Skill Match : {s.skill_match:.2f}%\n"
        f"Final Hiring Score : {s.final_score:.2f}%\n"
        f"Matched Skills : {', '.join(s.matched_skills) if s.matched_skills else 'None'}\n"
        f"Missing Skills : {', '.join(s.missing_skills) if s.missing_skills else 'None'}\n"
    )


# -------------------------------------------------
# Build Final Downloadable Report
# -------------------------------------------------
def build_report_text():
    s = st.session_state
    report = f"""
AI POWERED INTELLIGENT HIRING ASSISTANT
CANDIDATE REPORT
============================================================

Candidate Category    : {s.predicted_category}

Resume Similarity     : {s.similarity_score:.2f}%
Skill Match            : {s.skill_match:.2f}%
Final Hiring Score     : {s.final_score:.2f}%

Matched Skills:
{', '.join(s.matched_skills) if s.matched_skills else 'None'}

Missing Skills:
{', '.join(s.missing_skills) if s.missing_skills else 'None'}

------------------------------------------------------------
Rule Based Feedback
------------------------------------------------------------
{s.rule_feedback}

------------------------------------------------------------
AI Generated Feedback
------------------------------------------------------------
{s.ai_feedback if s.ai_feedback else 'Not generated yet.'}

============================================================
"""
    return report


# -------------------------------------------------
# NEW: Small helper that makes any text safe to put inside a PDF.
# It escapes special characters (like < > &) and turns line breaks
# into real PDF line breaks, so the AI feedback text doesn't break the PDF.
# -------------------------------------------------
def safe_pdf_text(text):
    text = escape(text)              # escape special characters for reportlab
    text = text.replace("\n", "<br/>")
    return text


# -------------------------------------------------
# NEW: Build Final Downloadable Report as a PDF file (in memory)
# Uses the reportlab library. Returns PDF bytes ready for st.download_button.
# -------------------------------------------------
def build_report_pdf():
    s = st.session_state

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []  # list of PDF elements (paragraphs, spacers, etc.)

    # Title
    story.append(Paragraph("AI Powered Intelligent Hiring Assistant", styles["Title"]))
    story.append(Paragraph("Candidate Report", styles["Heading2"]))
    story.append(Spacer(1, 14))

    # Core scores
    story.append(Paragraph(f"Candidate Category: {safe_pdf_text(str(s.predicted_category))}", styles["Normal"]))
    story.append(Paragraph(f"Resume Similarity: {s.similarity_score:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Skill Match: {s.skill_match:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Final Hiring Score: {s.final_score:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 14))

    # Matched skills
    story.append(Paragraph("Matched Skills", styles["Heading3"]))
    matched_text = ', '.join(s.matched_skills) if s.matched_skills else 'None'
    story.append(Paragraph(safe_pdf_text(matched_text), styles["Normal"]))
    story.append(Spacer(1, 10))

    # Missing skills
    story.append(Paragraph("Missing Skills", styles["Heading3"]))
    missing_text = ', '.join(s.missing_skills) if s.missing_skills else 'None'
    story.append(Paragraph(safe_pdf_text(missing_text), styles["Normal"]))
    story.append(Spacer(1, 14))

    # Rule based feedback
    story.append(Paragraph("Rule Based Feedback", styles["Heading3"]))
    story.append(Paragraph(safe_pdf_text(s.rule_feedback), styles["Normal"]))
    story.append(Spacer(1, 14))

    # AI generated feedback
    story.append(Paragraph("AI Generated Feedback", styles["Heading3"]))
    ai_feedback_text = s.ai_feedback if s.ai_feedback else "Not generated yet."
    story.append(Paragraph(safe_pdf_text(ai_feedback_text), styles["Normal"]))

    # Build the PDF into the in-memory buffer
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# -------------------------------------------------
# Small helper to draw a colored hero banner on each page
# -------------------------------------------------
def hero_banner(title, subtitle):
    st.markdown(f"""
    <div style="background: linear-gradient(90deg,#6C63FF,#8F87FF); padding:1.5rem 2rem;
                border-radius:14px; margin-bottom:1.5rem;">
        <h1 style="color:white; margin:0; font-size:1.8rem;">{title}</h1>
        <p style="color:#EDEBFF; margin:0.3rem 0 0 0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# Initialize Session State Variables
# -------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_feedback" not in st.session_state:
    st.session_state.ai_feedback = None
if "page" not in st.session_state:
    st.session_state.page = "analyzer"


# -------------------------------------------------
# Sidebar: Logo, Project Name, Developer, Tech Stack, Navigation
# -------------------------------------------------
with st.sidebar:

    # Project Logo + Name + Developer Credit
    st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size:3rem; line-height:1;">🎯</div>
            <div style="font-size:1.35rem; font-weight:700; margin-top:0.4rem;">AI Hiring Assistant</div>
            <div style="font-size:0.85rem; color:#B5B2D6; margin-top:0.2rem;">Developed by Devesh Sain</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#3a3d52; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown(
        "<div style='font-size:0.8rem; font-weight:600; color:#B5B2D6; letter-spacing:1px;'>TECHNOLOGY STACK</div>",
        unsafe_allow_html=True
    )
    tech_stack = ["🐍 Python", "⚡ Streamlit", "🧠 Scikit-learn", "🔤 Sentence-Transformers",
                  "🤖 GENAI",]
    badges_html = "".join(
        f"<span style='display:inline-block; background:#2b2e42; color:#e8e8ec; padding:4px 10px;"
        f"border-radius:12px; font-size:0.72rem; margin:4px 4px 0 0;'>{item}</span>"
        for item in tech_stack
    )
    st.markdown(f"<div style='margin-top:0.4rem;'>{badges_html}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#3a3d52; margin: 1rem 0 1rem 0;'>", unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='font-size:0.8rem; font-weight:600; color:#B5B2D6; letter-spacing:1px;'>NAVIGATION</div>",
        unsafe_allow_html=True
    )
    st.write("")

    if st.button("📊  Resume Analyzer", use_container_width=True,
                 type="primary" if st.session_state.page == "analyzer" else "secondary"):
        st.session_state.page = "analyzer"
        st.rerun()

    if st.button("🤖  AI Chatbot", use_container_width=True,
                 type="primary" if st.session_state.page == "chatbot" else "secondary"):
        st.session_state.page = "chatbot"
        st.rerun()


# -------------------------------------------------
# Load Models (with error handling)
# -------------------------------------------------
try:
    classifier, tfidf, encoder = load_models()
    embedding_model = load_embedding_model()
    knowledge_df = load_knowledge_base()

    # NEW: Build the RAG pipeline pieces ONCE (both are cached, so this is
    # instant on every rerun after the very first load).
    kb_documents = create_kb_documents(knowledge_df)
    kb_embeddings = create_kb_embeddings(embedding_model, kb_documents)

    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(
        "Required files not found. Please make sure resume_classifier.pkl, tfidf.pkl, "
        f"label_encoder.pkl and knowledge_base.csv are in the same folder as app.py.\n\nError: {e}"
    )


# ===================================================
# PAGE 1: RESUME ANALYZER
# ===================================================
if st.session_state.page == "analyzer":

    hero_banner(
        "🎯 AI Powered Intelligent Hiring Assistant",
        "Upload a candidate resume and a job description to evaluate the fit instantly."
    )

    # ---------------- Module 1: Resume + JD Upload ----------------
    st.header("1. Upload Resume and Job Description")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    with col2:
        jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

    analyze_button = st.button("Analyze Resume", type="primary")

    # ---------------- Run Full Pipeline on Button Click ----------------
    if analyze_button:
        if not models_loaded:
            st.error("Cannot run analysis because required model files are missing.")
        elif resume_file is None or jd_file is None:
            st.warning("Please upload both Resume and Job Description PDF files.")
        else:
            with st.spinner("Analyzing resume, please wait..."):
                # Extract text from both PDFs
                resume_text = extract_text_from_pdf(resume_file)
                jd_text = extract_text_from_pdf(jd_file)

                # Clean text
                clean_resume_text = clean_resume(resume_text)
                clean_jd_text = clean_resume(jd_text)

                # Resume Classification
                predicted_category = predict_category(clean_resume_text, classifier, tfidf, encoder)

                # Resume - JD Similarity
                similarity_score = get_similarity_score(clean_resume_text, clean_jd_text, embedding_model)

                # Skill Extraction
                resume_skills = extract_skills(clean_resume_text)
                jd_skills = extract_skills(clean_jd_text)
                matched_skills, missing_skills, extra_skills, skill_match = compare_skills(resume_skills, jd_skills)

                # Final Hiring Score
                final_score = calculate_final_score(skill_match, similarity_score)

                # Rule Based Feedback
                level, rule_feedback = rule_based_feedback(final_score, missing_skills)

                # Knowledge Base Context (used later for AI feedback / chatbot)
                knowledge_context = get_knowledge_context(missing_skills, knowledge_df)

                # Save everything to session state
                st.session_state.predicted_category = predicted_category
                st.session_state.similarity_score = similarity_score
                st.session_state.matched_skills = matched_skills
                st.session_state.missing_skills = missing_skills
                st.session_state.extra_skills = extra_skills
                st.session_state.skill_match = skill_match
                st.session_state.final_score = final_score
                st.session_state.level = level
                st.session_state.rule_feedback = rule_feedback
                st.session_state.knowledge_context = knowledge_context
                st.session_state.ai_feedback = None
                st.session_state.chat_history = []
                st.session_state.analyzed = True

            st.success("Analysis Complete!")

    st.markdown("---")

    # ---------------- Show Results only if analysis has been done ----------------
    if st.session_state.analyzed:

        # Module 2: Resume Classification
        st.header("2. Resume Classification")
        st.metric("Predicted Category", st.session_state.predicted_category)
        st.markdown("---")

        # Module 3: Resume Similarity
        st.header("3. Resume - JD Similarity")
        st.write(f"Similarity Score: {st.session_state.similarity_score:.2f}%")
        st.progress(min(int(st.session_state.similarity_score), 100))
        st.markdown("---")

        # Module 4: Skill Gap Analysis
        st.header("4. Skill Gap Analysis")
        col1, col2 = st.columns(2)

        with col1:
            with st.expander(f"✔ Matched Skills ({len(st.session_state.matched_skills)})", expanded=True):
                if st.session_state.matched_skills:
                    for skill in st.session_state.matched_skills:
                        st.write("✔", skill)
                else:
                    st.write("No matched skills found.")

        with col2:
            with st.expander(f"❌ Missing Skills ({len(st.session_state.missing_skills)})", expanded=True):
                if st.session_state.missing_skills:
                    for skill in st.session_state.missing_skills:
                        st.write("❌", skill)
                else:
                    st.write("No missing skills. Great job!")

        with st.expander(f"⭐ Additional Skills ({len(st.session_state.extra_skills)})"):
            if st.session_state.extra_skills:
                for skill in st.session_state.extra_skills:
                    st.write("⭐", skill)
            else:
                st.write("No extra skills found.")

        st.write(f"Skill Match: {st.session_state.skill_match:.2f}%")
        st.progress(min(int(st.session_state.skill_match), 100))
        st.markdown("---")

        # Module 5: Final Hiring Score
        st.header("5. Final Hiring Score")
        col1, col2, col3 = st.columns(3)
        col1.metric("Resume Similarity", f"{st.session_state.similarity_score:.2f}%")
        col2.metric("Skill Match", f"{st.session_state.skill_match:.2f}%")
        col3.metric("Final Hiring Score", f"{st.session_state.final_score:.2f}%")
        st.progress(min(int(st.session_state.final_score), 100))
        st.markdown("---")

        # Module 6: Rule Based Feedback
        st.header("6. Rule Based Feedback")
        with st.container(border=True):
            if st.session_state.level == "Excellent Match":
                st.success(st.session_state.rule_feedback)
            elif st.session_state.level == "Good Match":
                st.info(st.session_state.rule_feedback)
            else:
                st.warning(st.session_state.rule_feedback)
        st.markdown("---")

        # Module 7: AI Feedback
        st.header("7. AI Generated Feedback")
        if st.button("Generate AI Feedback", type="primary"):
            with st.spinner("Generating AI feedback..."):
                try:
                    client = get_openrouter_client(API_KEY)
                    ai_feedback = generate_ai_feedback(
                        client,
                        st.session_state.predicted_category,
                        st.session_state.similarity_score,
                        st.session_state.skill_match,
                        st.session_state.matched_skills,
                        st.session_state.missing_skills,
                        st.session_state.knowledge_context
                    )
                    st.session_state.ai_feedback = ai_feedback
                except Exception as e:
                    st.error(f"Could not generate AI feedback. Please check your OpenRouter API key.\n\nError: {e}")

        if st.session_state.ai_feedback:
            with st.container(border=True):
                st.markdown(st.session_state.ai_feedback)
        st.markdown("---")

        # Module 8: Download Report
        st.header("8. Download Report")
        report_pdf = build_report_pdf()   # UPDATED: now builds a PDF instead of plain text
        st.download_button(
            label="Download Candidate Report",
            data=report_pdf,
            file_name="candidate_report.pdf",
            mime="application/pdf"
        )

    else:
        st.info("Upload a resume and job description, then click 'Analyze Resume' to begin.")


# ===================================================
# PAGE 2: AI CHATBOT
# ===================================================
elif st.session_state.page == "chatbot":

    hero_banner(
        "🤖 AI Chatbot",
        "Ask questions about your resume analysis, missing skills, or interview tips."
    )

    if not st.session_state.analyzed:
        st.info("Please analyze a resume first from the 📊 Resume Analyzer page.")
    else:
        # Show previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Type your question here...")

        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            analysis_context = build_analysis_context()

            # ---- NEW: RAG STEP - retrieve Top-5 relevant knowledge base rows ----
            retrieved_docs = retrieve_relevant_knowledge(
                user_question, embedding_model, kb_embeddings, kb_documents, top_k=5
            )
            retrieved_context = build_rag_context(retrieved_docs)

            try:
                client = get_openrouter_client(API_KEY)
                with st.spinner("Thinking..."):
                    reply = get_chatbot_reply(
                        client, user_question, analysis_context,
                        retrieved_context, st.session_state.chat_history
                    )

                with st.chat_message("assistant"):
                    st.write(reply)

                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Chatbot is unavailable right now. Please check your OpenRouter API key.\n\nError: {e}")
