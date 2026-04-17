import streamlit as st
import pickle

st.set_page_config(
    page_title="Phishing Detector",
    page_icon="📧",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

with open("model/phishing_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

PHISHING_KEYWORDS = [
    "urgent",
    "verify",
    "password",
    "login",
    "account",
    "suspended",
    "click here",
    "bank",
    "security",
    "immediately",
    "action required",
    "limited time"
]

def keyword_score(text):
    text = text.lower()
    score = 0
    for word in PHISHING_KEYWORDS:
        if word in text:
            score += 0.08
    return min(score, 0.4)

def highlight_keywords(text):
    words = text.lower().split()
    flagged = []
    for w in words:
        for k in PHISHING_KEYWORDS:
            if k in w:
                flagged.append(w)
                break
    return flagged

st.title("📧 Phishing Email Detector")
st.write("Check if an email is safe or phishing using AI")

email_text = st.text_area("Paste email content here")

if st.button("Check"):

    if email_text.strip():

        ml_prob = model.predict_proba([email_text])[0][1]
        kw_boost = keyword_score(email_text)
        prob = min(ml_prob + kw_boost, 1.0)

        st.markdown("---")
        st.subheader("Result")

        if prob > 0.5:
            label = "⚠️ Phishing"
            confidence = prob
            st.error("This email is likely a phishing attempt.")
        else:
            label = "✅ Safe"
            confidence = 1 - prob
            st.success("This email appears safe.")

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", label)
        col2.metric("Confidence", f"{confidence:.2%}")
        col3.metric("Risk Score", f"{prob*100:.1f}")

        st.progress(int(confidence * 100))

        st.markdown("### 🔍 Why this was flagged")

        flags = highlight_keywords(email_text)

        if flags:
            st.write("Suspicious words detected:")
            st.write(", ".join(flags))
        else:
            st.write("No strong phishing keywords detected")

    else:
        st.warning("Please enter email content")
        