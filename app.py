import streamlit as st
from transformers import pipeline

# Load model (using Hugging Face pipeline for paraphrasing)
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

paraphraser = load_model()

# Streamlit UI
st.set_page_config(page_title="Essay Rephraser AI", page_icon="📝", layout="wide")

st.title("📝 Generative AI Essay Rephraser")
st.write("Paste your essay below and get a plagiarism-free rephrased version.")

# Input text
input_text = st.text_area("Enter your essay:", height=250)

if st.button("Rephrase ✨"):
    if input_text.strip():
        with st.spinner("Rephrasing your essay..."):
            result = paraphraser(
                f"paraphrase: {input_text}",
                max_length=512,
                num_return_sequences=1,
                temperature=0.9,
                top_p=0.95,
                do_sample=True
            )
            st.subheader("✅ Rephrased Essay:")
            st.write(result[0]['generated_text'])
    else:
        st.warning("⚠️ Please enter some text before rephrasing.")

# st.caption("Built with Hugging Face Transformers + Streamlit")
