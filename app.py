import streamlit as st
import pandas as pd
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="College Admission Chatbot", page_icon="🎓", layout="wide")

# ==========================================
# 11. UI Design: Custom CSS Theme (Blue & White)
# ==========================================
st.markdown("""
<style>
/* Global Dark Theme */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* Headings */
h1, h2, h3 {
    color: #38bdf8;
    font-weight: 700;
}

/* Glass Card Effect */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Input Box */
.stTextInput input, .stChatInput textarea {
    background-color: #020617 !important;
    color: #e2e8f0 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(56,189,248,0.5);
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
}

/* Alerts */
.success-box {
    padding: 10px;
    border-radius: 10px;
    background: rgba(34,197,94,0.1);
    border: 1px solid #22c55e;
}

.error-box {
    padding: 10px;
    border-radius: 10px;
    background: rgba(239,68,68,0.1);
    border: 1px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)


DATA_FILE = "DATA.json"
MODELS_DIR = "models"

# ==========================================
# 1. Dataset Handling
# ==========================================
@st.cache_data
def load_and_prepare_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Handle the structure of the dataset (Flatten intents if it's in intent format)
    if isinstance(data, dict) and "intents" in data:
        records = []
        for intent in data["intents"]:
            tag = intent["tag"]
            responses = intent.get("responses", [""])
            response = responses[0] if responses else ""
            for pattern in intent.get("patterns", []):
                records.append({
                    "question": pattern,
                    "answer": response,
                    "category": tag
                })
        df = pd.DataFrame(records)
    elif isinstance(data, list):
        # Handle simple flat JSON list
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()
    return df

# ==========================================
# 4. Functions & NLP
# ==========================================
def preprocess_text(text):
    """Clean and preprocess string."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# ==========================================
# 2. & 3. Machine Learning and Auto Logic
# ==========================================
@st.cache_resource
def load_models():
    """Load models if available, otherwise train them."""
    required_files = ["model1.pkl", "model2.pkl", "model3.pkl", "vectorizer.pkl", "stats.json"]
    missing = False
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        missing = True
    else:
        for f in required_files:
            if not os.path.exists(os.path.join(MODELS_DIR, f)):
                missing = True
                break
                
    if missing:
        # Auto Train models since they are missing
        df = load_and_prepare_data()
        if df.empty:
            return None, None, None, None, None
            
        df['processed_question'] = df['question'].apply(preprocess_text)
        X = df['processed_question']
        y = df['category']
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        
        # Model A: Logistic Regression
        model1 = LogisticRegression(max_iter=1000)
        model1.fit(X_vec, y)
        acc1 = accuracy_score(y, model1.predict(X_vec))
        
        # Model B: Multinomial Naive Bayes
        model2 = MultinomialNB()
        model2.fit(X_vec, y)
        acc2 = accuracy_score(y, model2.predict(X_vec))
        
        # Model C: Linear SVM
        model3 = LinearSVC(max_iter=1000)
        model3.fit(X_vec, y)
        acc3 = accuracy_score(y, model3.predict(X_vec))
        
        stats = {
            "Logistic Regression": acc1,
            "Multinomial NB": acc2,
            "Linear SVM": acc3
        }
        
        # Save models to directory
        joblib.dump(model1, os.path.join(MODELS_DIR, "model1.pkl"))
        joblib.dump(model2, os.path.join(MODELS_DIR, "model2.pkl"))
        joblib.dump(model3, os.path.join(MODELS_DIR, "model3.pkl"))
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
        with open(os.path.join(MODELS_DIR, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f)
            
    else:
        # Load Existing Models (Do NOT retrain)
        model1 = joblib.load(os.path.join(MODELS_DIR, "model1.pkl"))
        model2 = joblib.load(os.path.join(MODELS_DIR, "model2.pkl"))
        model3 = joblib.load(os.path.join(MODELS_DIR, "model3.pkl"))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
        with open(os.path.join(MODELS_DIR, "stats.json"), "r", encoding="utf-8") as f:
            stats = json.load(f)
            
    return model1, model2, model3, vectorizer, stats

# ==========================================
# 5. Ensemble Logic
# ==========================================
def get_ensemble_prediction(text):
    m1, m2, m3, vec, _ = load_models()
    if m1 is None:
        return None, None, None, None
        
    proc_text = preprocess_text(text)
    vec_text = vec.transform([proc_text])
    
    pred1 = m1.predict(vec_text)[0]
    pred2 = m2.predict(vec_text)[0]
    pred3 = m3.predict(vec_text)[0]
    
    preds = [pred1, pred2, pred3]
    # Majority Voting
    final_pred = max(set(preds), key=preds.count)
    
    return final_pred, pred1, pred2, pred3

# ==========================================
# 6. Chatbot Logic
# ==========================================
def get_best_answer(category, user_query):
    df = load_and_prepare_data()
    cat_df = df[df['category'] == category].copy()
    
    if cat_df.empty:
        return "I'm sorry, I don't have information about that right now."
        
    # Use TF-IDF cosine similarity to find the most similar question in the category
    vec = TfidfVectorizer()
    all_qs = cat_df['question'].tolist()
    all_qs.append(user_query)
    
    vec_matrix = vec.fit_transform(all_qs)
    cos_sim = cosine_similarity(vec_matrix[-1], vec_matrix[:-1]).flatten()
    best_idx = np.argmax(cos_sim)
    
    # If the similarity is too low, we fallback to just the default answer for that category
    if len(cos_sim) > 0 and cos_sim[best_idx] < 0.1:
        return cat_df.iloc[0]['answer']
        
    return cat_df.iloc[best_idx]['answer']

# ==========================================
# 7. PDF Chatbot Logic
# ==========================================
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extr = page.extract_text()
            if extr:
                text += extr + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"
    
def answer_from_pdf(pdf_text, query):
    if not pdf_text.strip():
        return None
        
    # Basic similarity search over sentences
    sentences = re.split(r'(?<=[.!?]) +', pdf_text)
    # filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return None
        
    sentences.append(query)
    vec = TfidfVectorizer().fit_transform(sentences)
    cos_sim = cosine_similarity(vec[-1], vec[:-1]).flatten()
    best_idx = np.argmax(cos_sim)
    
    # Threshold for matching
    if cos_sim[best_idx] < 0.05:
        return None 
        
    # Return surrounding context
    start = max(0, best_idx - 1)
    end = min(len(sentences), best_idx + 2)
    answer = " ".join(sentences[start:end])
    return answer

# ==========================================
# 8. Authentication
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    # Centered Login Box
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.title("🎓 Admission Portal Login")
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid credentials!")
else:
    # ==========================================
    # Main Application
    # ==========================================
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🎓 College Admission Chatbot")
    with col2:
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    m1, m2, m3, vec, stats = load_models()
    
    if m1 is None:
        st.error("Missing/Empty DATA.json. Cannot train models.")
        st.stop()
    
    # ==========================================
    # 10. Sidebar Dashboard
    # ==========================================
    with st.sidebar:
        st.title("Admin Dashboard ⚙️")
        st.markdown("### Model Performance")
        
        # Display performance table
        perf_data = []
        for model_name, acc in stats.items():
            perf_data.append({
                "Model": model_name,
                "Accuracy": f"{acc*100:.2f}%",
                "Error Rate": f"{(1-acc)*100:.2f}%"
            })
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, hide_index=True)
        
        st.markdown("### PDF Context Helper")
        st.write("Upload a PDF to allow the bot to search it for answers.")
        uploaded_pdf = st.file_uploader("Upload PDF File", type=['pdf'])
        
        if uploaded_pdf:
            if 'pdf_text' not in st.session_state or st.session_state.get('pdf_name') != uploaded_pdf.name:
                pdf_text = extract_pdf_text(uploaded_pdf)
                st.session_state['pdf_text'] = pdf_text
                st.session_state['pdf_name'] = uploaded_pdf.name
            st.success(f"PDF Context Active: {uploaded_pdf.name}")
        else:
            st.session_state['pdf_text'] = None
            st.session_state['pdf_name'] = None

    # ==========================================
    # 9. & 12. Streamlit UI (Tabs)
    # ==========================================
    tab1, tab2 = st.tabs(["💬 AI Chatbot", "📊 College Advisor"])
    
    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Hello! I am your AI College Admission Assistant. Ask me anything!"
            })
            
        # Render Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat Input
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            with st.chat_message("assistant"):
                final_response = ""
                
                # Check PDF first if available
                pdf_answered = False
                if st.session_state.get('pdf_text'):
                    pdf_ans = answer_from_pdf(st.session_state['pdf_text'], user_input)
                    if pdf_ans:
                        final_response += f"📄 **Excerpt from PDF:**\n> {pdf_ans}\n\n"
                        pdf_answered = True
                
                # Predict from Models
                final_pred, p1, p2, p3 = get_ensemble_prediction(user_input)
                best_ans = get_best_answer(final_pred, user_input)
                
                # Append Chatbot DB Response
                if best_ans:
                    final_response += f"🤖 **Answer:** {best_ans}\n\n"
                
                # 12. Must include debugging information
                final_response += f"""---
**Diagnostic Data:**
- Final Predicted Category: `{final_pred}`
- Model A (LR) Prediction: `{p1}`
- Model B (NB) Prediction: `{p2}`
- Model C (SVM) Prediction: `{p3}`
"""
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

    with tab2:
        st.header("🎯 AI College Advisor")
        st.markdown("Find the best colleges based on your profile using Probabilistic Modeling.")
        st.markdown("---")
        
        if not os.path.exists("colleges.csv"):
            st.error("colleges.csv not found!")
        else:
            df_col = pd.read_csv("colleges.csv")
            
            c1, c2 = st.columns(2)
            with c1:
                student_marks = st.number_input("Enter Marks / Percentile (0 - 100)", min_value=0.0, max_value=100.0, value=80.0)
                student_budget = st.number_input("Max Budget (₹)", min_value=0, value=250000, step=10000)
            with c2:
                locations = df_col['location'].unique().tolist()
                branches = df_col['branch'].unique().tolist()
                student_loc = st.selectbox("Preferred Location", ["Any"] + locations)
                student_branch = st.selectbox("Preferred Branch", ["Any"] + branches)
                
            if st.button("🚀 Get Advisor Report", use_container_width=True):
                # Filter Dataset
                if student_loc != "Any":
                    df_col = df_col[df_col['location'] == student_loc]
                if student_branch != "Any":
                    df_col = df_col[df_col['branch'] == student_branch]
                df_col = df_col[df_col['fees'] <= student_budget]
                
                if df_col.empty:
                    st.warning("No colleges found matching your criteria. Try adjusting your constraints.")
                else:
                    st.markdown("### 🏆 Top Recommended Colleges")
                    import math
                    chances = []
                    for _, row in df_col.iterrows():
                        diff = student_marks - row['avg_cutoff']
                        # Logistic regression sigmoid curve mapping diff to probability
                        prob = 1 / (1 + math.exp(-0.35 * diff))
                        chances.append(min(99.9, max(1.0, prob * 100)))
                        
                    df_col['Admission Chance (%)'] = chances
                    df_col['Admission Chance (%)'] = df_col['Admission Chance (%)'].round(2)
                    
                    # Sort by chance
                    df_col = df_col.sort_values(by='Admission Chance (%)', ascending=False)
                    
                    # Visual enhancements
                    def color_chance(val):
                        if val >= 80:
                            color = '#22c55e' # green
                        elif val >= 50:
                            color = '#eab308' # yellow
                        else:
                            color = '#ef4444' # red
                        return f'color: {color}; font-weight: bold;'
                        
                    st.dataframe(
                        df_col[['college', 'location', 'branch', 'fees', 'avg_cutoff', 'Admission Chance (%)']].style.map(color_chance, subset=['Admission Chance (%)']),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("### 📈 Cutoff Comparison vs Your Score")
                    # Prepare comparison chart
                    chart_data = df_col[['college', 'avg_cutoff']].set_index('college')
                    chart_data['Your Score'] = student_marks
                    st.bar_chart(chart_data)
