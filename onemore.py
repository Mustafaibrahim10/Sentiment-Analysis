import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('final_bidirectional_gru_model.keras')
tokenizer = joblib.load('tokenizer.pkl')

max_len = 100  # Must match training input length

# ------------------ Custom CSS ------------------ #
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            font-size: 40px;
            color: #3b5998;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            font-size: 20px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .result-box {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ App UI ------------------ #

st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by a Bidirectional GRU Neural Network</div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("📝 Enter your movie review below:", height=200)

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid review to analyze.")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

        # Prediction
        prediction = model.predict(padded)[0][0]
        sentiment = "😊 Positive" if prediction >= 0.5 else "😞 Negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        # Output
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: **{confidence:.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)