import streamlit as st
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()  # loads environment variables from .env

# Load your custom FAQ
with open("faq.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()

faq_sections = [section.strip() for section in faq_text.split("\n\n") if section.strip()]

# Fallback to TF-IDF instead of SentenceTransformer for Python 3.14 compatibility
tfidf_vectorizer = TfidfVectorizer()
faq_embeddings = tfidf_vectorizer.fit_transform(faq_sections)


def find_most_relevant_section(query, faq_sections, faq_embeddings):
    query_embedding = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return faq_sections[best_match_idx]


def ask_gpt(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# Streamlit UI
st.title("📚 Smart FAQ Chatbot")
st.write("Ask a question about our FAQ and get an instant answer.")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Searching for the best answer..."):
        context = find_most_relevant_section(user_question, faq_sections, faq_embeddings)
        answer = ask_gpt(context, user_question)
        st.markdown("### ✅ Answer:")
        st.write(answer)
        st.markdown("---")
        st.markdown("**Matched FAQ Section:**")
        st.code(context)

st.caption("Built by Kristian Gichev")
