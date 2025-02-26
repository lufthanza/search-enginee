import streamlit as st
import sqlite3
import fitz  # PyMuPDF for PDF processing
import docx
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import re

# 📌 Load Model SBERT dari Hugging Face
@st.cache_resource
def load_sbert_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, sbert_model = load_sbert_model()

# 📌 Load Model DPR dari Hugging Face
@st.cache_resource
def load_dpr_model():
    dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    dpr_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    return dpr_tokenizer, dpr_model

dpr_tokenizer, dpr_model = load_dpr_model()

# 📌 Fungsi ekstraksi teks dari file
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text") + " "
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    return text.strip()

# 📌 Simpan file ke database
def store_file(filename, content):
    conn = sqlite3.connect("files_db.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS files (filename TEXT PRIMARY KEY, content TEXT)")
    cursor.execute("INSERT OR REPLACE INTO files (filename, content) VALUES (?, ?)", (filename, content))
    conn.commit()
    conn.close()

# 📌 Proses upload file
def process_files(uploaded_files):
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        store_file(uploaded_file.name, text)

# 📌 Exact Match Search
def exact_match_search(keyword):
    conn = sqlite3.connect("files_db.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM files")
    results = []
    for filename, content in cursor.fetchall():
        sentences = content.split(". ")
        matches = [s for s in sentences if keyword.lower() in s.lower()]
        if matches:
            results.append((filename, [(s, 100.0) for s in matches]))
    conn.close()
    return results

# 📌 SBERT Semantic Search
def sbert_search(keyword):
    conn = sqlite3.connect("files_db.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM files")
    data = cursor.fetchall()
    conn.close()

    if not data:
        return []

    filenames, documents = zip(*data)
    document_embeddings = encode_texts(documents)
    keyword_embedding = encode_texts([keyword])[0]
    similarities = np.dot(document_embeddings, keyword_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(keyword_embedding)
    )

    results = []
    for idx, score in enumerate(similarities):
        if score > 0.5:
            sentences = documents[idx].split(". ")
            matches = [s for s in sentences if keyword.lower() in s.lower()]
            if matches:
                results.append((filenames[idx], [(m, score * 100) for m in matches]))
    return results

# 📌 Encode teks menggunakan SBERT
def encode_texts(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = sbert_model(**tokens).last_hidden_state[:, 0, :]
    return embeddings.numpy()

# 📌 DPR Search
def dpr_search(keyword):
    conn = sqlite3.connect("files_db.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM files")
    data = cursor.fetchall()
    conn.close()

    if not data:
        return []

    filenames, documents = zip(*data)

    # Encode dokumen dan query
    document_embeddings = encode_dpr_texts(documents)
    query_embedding = encode_dpr_texts([keyword])[0]  # Pastikan query berbentuk vektor 1D

    # Hitung kesamaan kosinus
    similarities = np.dot(document_embeddings, query_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    results = []
    for idx, score in enumerate(similarities):
        if score > 0.5:  # Filter hasil yang relevan
            sentences = documents[idx].split(". ")
            matches = [s for s in sentences if keyword.lower() in s.lower()]
            if matches:
                results.append((filenames[idx], [(m, score * 100) for m in matches]))

    return results


# 📌 Encode teks menggunakan DPR
def encode_dpr_text(text):
    tokens = dpr_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embedding = dpr_model(**tokens).pooler_output
    return embedding.numpy()

# 📌 Encode teks menggunakan DPR (Dense Passage Retrieval)
def encode_dpr_texts(texts):
    """Mengubah daftar teks menjadi embedding DPR dengan batas maksimal 512 token"""
    embeddings = []
    for text in texts:
        tokens = dpr_tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            embedding = dpr_model(**tokens).pooler_output  # Mengambil embedding akhir
        embeddings.append(embedding.numpy())
    return np.vstack(embeddings)  # Susun menjadi array 2D


# 📌 UI Streamlit
st.title("🔍 Search Engine dengan BM25, SBERT, DPR, dan Exact Match")

uploaded_files = st.file_uploader("📂 Unggah file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    process_files(uploaded_files)
    st.success("✅ Semua file berhasil disimpan dalam database.")

search_method = st.selectbox("📌 Pilih metode pencarian:", ["Exact Match", "SBERT", "DPR"])
keyword = st.text_input("🔎 Masukkan kata kunci:")

if st.button("Cari") and keyword:
    results = []
    if search_method == "Exact Match":
        results = exact_match_search(keyword)
    elif search_method == "SBERT":
        results = sbert_search(keyword)
    elif search_method == "DPR":
        results = dpr_search(keyword)

    if results:
        for filename, matches in results:
            st.subheader(f"📁 Hasil dalam: {filename}")
            for sentence, similarity in matches:
                st.write(f"- {sentence} (🔹 Relevansi: {similarity:.2f}%)")
    else:
        st.warning("❌ Tidak ada hasil yang cocok.")
