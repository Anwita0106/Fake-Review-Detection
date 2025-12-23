import streamlit as st
import pickle
import re
import time

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🛒",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #1e3a8a);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
}

/* Gradient animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism card */
.main {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    padding: 38px;
    border-radius: 22px;
    box-shadow: 0px 30px 60px rgba(0,0,0,0.45);
    max-width: 880px;
    margin: auto;
    animation: fadeUp 0.8s ease;
}

/* Entry animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(25px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Text colors */
h1, h2, h3, label, p {
    color: #e5e7eb !important;
}

/* Text area */
textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 14px !important;
    font-size: 16px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 16px;
    height: 54px;
    font-size: 18px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0px 10px 25px rgba(99,102,241,0.5);
}

/* Result cards */
.result-card {
    padding: 22px;
    border-radius: 16px;
    margin-top: 22px;
    font-size: 18px;
    animation: fadeUp 0.6s ease;
}

.fake {
    background: rgba(239, 68, 68, 0.15);
    border-left: 6px solid #ef4444;
}

.genuine {
    background: rgba(34, 197, 94, 0.15);
    border-left: 6px solid #22c55e;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 36px;
    color: #cbd5f5;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)







# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------- UI ----------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("## 🛒 Fake Product Review Detection System")
st.write("✨ Detect whether a product review is **Fake or Genuine** using **Machine Learning & NLP**.")

review = st.text_area(
    "✍️ Enter Product Review",
    height=150,
    placeholder="Example: This product is amazing, totally worth the price!"
)

if st.button("🔍 Analyze Review"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        with st.spinner("🤖 Analyzing review using AI..."):
            time.sleep(1.5)

            cleaned = clean_text(review)
            vect = vectorizer.transform([cleaned])
            prediction = model.predict(vect)[0]
            confidence = model.predict_proba(vect)[0][prediction]

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card genuine">
                ✅ <b>Genuine Review</b><br>
                Confidence: {confidence:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(confidence)
        else:
            st.markdown(
                f"""
                <div class="result-card fake">
                🚨 <b>Fake Review Detected</b><br>
                Confidence: {confidence:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(confidence)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
🚀 Built using Machine Learning, NLP & Streamlit<br>
For real-world e-commerce fraud detection
</div>
""", unsafe_allow_html=True)
