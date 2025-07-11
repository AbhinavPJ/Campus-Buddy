import streamlit as st
import torch
import requests
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from rank_bm25 import BM25Okapi
import time
import openai
st.set_page_config(page_title="🧑‍🎓 Senior Chatbot", layout="wide")

# === Load Models ===
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_data():
    if not os.path.exists("hybrid_embeddings.pt") or not os.path.exists("bm25_index.pkl"):
        st.error("❌ Required files not found. Run `generate_embeddings.py` first.")
        st.stop()
    
    bert_data = torch.load("hybrid_embeddings.pt")
    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    
    return (
        bert_data["passages"],
        bert_data["embeddings"],
        bm25_data["bm25"],
        bm25_data["bm25_tokens"]
    )

# === Utility Functions ===
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def preprocess(text):
    return re.sub(r"[^\w\s]", " ", text).lower().split()

def hybrid_retrieve(query, tokenizer, model, passages, passage_embeddings, bm25, bm25_tokens, k=3):
    q_embed = get_embedding(query, tokenizer, model)
    q_tokens = preprocess(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bert_scores = cosine_similarity(q_embed, passage_embeddings.numpy())[0]
    bm25_norm = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
    bert_norm = (bert_scores + 1) / 2
    hybrid = 0.3 * bm25_norm + 0.7 * bert_norm
    
    top_k_idx = np.argsort(hybrid)[-k:][::-1]
    return [passages[i] for i in top_k_idx]

def ask_llama(passages, query):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY not set."

    try:
        openai.api_key = api_key
        openai.api_base = "https://api.groq.com/openai/v1"

        prompt = (
            "You are a friendly and smart college senior helping a junior with questions.\n"
            "Be informal, clear, and explain things simply. Don’t say things like 'based on the context'.\n\n"
            f"Context:\n{''.join(passages)}\n\nQuestion: {query}"
        )

        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful, informal college senior."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=1024,
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        return f"❌ Error: {str(e)}"




# === UI ===
st.title("🧑‍🎓 Ask Your College Senior")
st.caption("A simple chatbot that uses your documents to answer like a chill senior.")
tokenizer, model = load_bert()
passages, passage_embeddings, bm25, bm25_tokens = load_data()
query = st.text_input("Ask me anything:", placeholder="e.g., Does CGPA really matter?")

if query:
    with st.spinner("Thinking like your cool senior..."):
        top_passages = hybrid_retrieve(query, tokenizer, model, passages, passage_embeddings, bm25, bm25_tokens)
        answer = ask_llama(top_passages, query)
    
    st.markdown("### 🧑‍🏫 Here's what I think:")
    st.markdown(answer)
