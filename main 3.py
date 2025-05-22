import os
import re
import nltk
import streamlit as st
import sqlite3
import hashlib
import pickle
import datetime
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.translate.meteor_score import meteor_score
from collections import Counter

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Add Indonesian stopwords
indonesian_stop_words = {
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", 
    "akhir", "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", 
    "antar", "antara", "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", 
    "apatah", "artinya", "asal", "asalkan", "atas", "atau", "ataukah", "ataupun", 
    "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", 
    "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", 
    "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini", "beginian", 
    "beginikah", "beginilah", "begitu", "begitukah", "begitulah", "begitupun", 
    "bekerja", "belakang", "belakangan", "belum", "belumlah", "benar", "benarkah", 
    "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya", "berapa", 
    "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", "berdatangan", 
    "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali-kali", "berkata", 
    "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", 
    "berlebihan", "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", 
    "bersama-sama", "bersiap", "bersiap-siap", "bertanya", "bertanya-tanya", "berturut", 
    "berturut-turut", "bertutur", "berujar", "berupa", "besar", "betul", "betulkah", 
    "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "boleh", "bolehkah", 
    "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung", 
    "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", 
    "dan", "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian", 
    "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", 
    "diantara", "diantaranya", "diberi", "diberikan", "diberikannya", "dibuat", 
    "dibuatnya", "didapat", "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", 
    "diingat", "diingatkan", "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", 
    "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui", "diketahuinya", 
    "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", 
    "dimaksudnya", "diminta", "dimintai", "dimisalkan", "dimulai", "dimulailah", 
    "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", "diperbuatnya", 
    "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya", 
    "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan", 
    "disebut", "disebutkan", "disebutkannya", "disini", "disinilah", "ditambahkan", 
    "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", 
    "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", 
    "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua", "dulu", 
    "empat", "enggak", "enggaknya", "entah", "entahlah", "guna", "gunakan", "hal", 
    "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah", "harusnya", "hendak", 
    "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", 
    "ibu", "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", "ini", 
    "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", 
    "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", 
    "jelas", "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", 
    "jumlahnya", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", 
    "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "karena", 
    "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", 
    "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan", 
    "kelihatannya", "kelima", "keluar", "kembali", "kemudian", "kemungkinan", 
    "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesamaan", "keseluruhan", 
    "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira", 
    "kira-kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", 
    "lah", "lain", "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya", 
    "lebih", "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah", 
    "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah", 
    "masalahnya", "masih", "masihkah", "masing", "masing-masing", "mau", "maupun", 
    "melainkan", "melakukan", "melalui", "melihat", "melihatnya", "memang", "memastikan", 
    "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta", "memintakan", 
    "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan", 
    "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", 
    "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti", "menanti-nanti", 
    "menantikan", "menanya", "menanyai", "menanyakan", "mendapat", "mendapatkan", 
    "mendatang", "mendatangi", "mendatangkan", "menegaskan", "mengakhiri", "mengapa", 
    "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui", "menggunakan", 
    "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan", 
    "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", 
    "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", 
    "menunjukkan", "menunjuknya", "menurut", "menuturkan", "menyampaikan", "menyangkut", 
    "menyatakan", "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka", 
    "merekalah", "merupakan", "meski", "meskipun", "meyakini", "meyakinkan", "minta", 
    "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", 
    "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris", 
    "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya", "pak", "paling", 
    "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya", "per", 
    "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", 
    "pertama-tama", "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula", 
    "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja", 
    "sajalah", "saling", "sama", "sama-sama", "sambil", "sampai", "sampai-sampai", 
    "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", 
    "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", 
    "sebaik-baiknya", "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebegitu", 
    "sebelum", "sebelumnya", "sebenarnya", "seberapa", "sebesar", "sebetulnya", 
    "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara", "secukupnya", 
    "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala", 
    "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh", 
    "sejenak", "sejumlah", "sekadar", "sekadarnya", "sekali", "sekali-kali", "sekalian", 
    "sekaligus", "sekalipun", "sekarang", "sekarang", "sekecil", "seketika", "sekiranya", 
    "sekitar", "sekitarnya", "sekurang-kurangnya", "sekurangnya", "sela", "selain", 
    "selaku", "selalu", "selama", "selama-lamanya", "selamanya", "selanjutnya", "seluruh", 
    "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih", 
    "semata", "semata-mata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", 
    "semua", "semuanya", "semula", "sendiri", "sendirian", "sendirinya", "seolah", 
    "seolah-olah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah", "seperlunya", 
    "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta", "serupa", 
    "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", 
    "sesuatunya", "sesudah", "sesudahnya", "setelah", "setempat", "setengah", "seterusnya", 
    "setiap", "setiba", "setibanya", "setidak-tidaknya", "setidaknya", "setinggi", 
    "seusai", "sewaktu", "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", 
    "soal", "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", 
    "tadinya", "tahu", "tahun", "tak", "tambah", "tambahnya", "tampak", "tampaknya", 
    "tandas", "tandasnya", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", 
    "tegasnya", "telah", "tempat", "tengah", "tentang", "tentu", "tentulah", "tentunya", 
    "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat", "terdiri", 
    "terhadap", "terhadapnya", "teringat", "teringat-ingat", "terjadi", "terjadilah", 
    "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", 
    "tersampaikan", "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama", 
    "tetap", "tetapi", "tiap", "tiba", "tiba-tiba", "tidak", "tidakkah", "tidaklah", 
    "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", 
    "ujar", "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah", 
    "usai", "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun", "wong", 
    "yaitu", "yakin", "yakni", "yang"
}
stop_words.update(indonesian_stop_words)

# Database constants
DB_PATH = "pdf_search.db"

def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_database():
    """Initialize database tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create files table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create chunks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        tokens BLOB NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files (id),
        UNIQUE (file_id, chunk_index)
    )
    ''')
    
    conn.commit()
    conn.close()

def has_database_content():
    """Check if database has any content"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    file_count = cursor.fetchone()[0]
    conn.close()
    return file_count > 0

def get_file_hash(file):
    """Generate hash for file to check if it's already in database"""
    file_content = file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def store_file_in_db(file, chunks, tokenized_chunks):
    """Store file and its chunks in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Reset file pointer
        file.seek(0)
        file_hash = get_file_hash(file)
        
        # Check if file already exists
        cursor.execute("SELECT id FROM files WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            # File already exists
            file_id = existing[0]
            # Delete existing chunks for this file
            cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        else:
            # Insert new file
            cursor.execute(
                "INSERT INTO files (filename, file_hash) VALUES (?, ?)",
                (file.name, file_hash)
            )
            file_id = cursor.lastrowid
        
        # Insert chunks
        for idx, (chunk, tokens) in enumerate(zip(chunks, tokenized_chunks)):
            cursor.execute(
                "INSERT INTO chunks (file_id, chunk_index, chunk_text, tokens) VALUES (?, ?, ?, ?)",
                (file_id, idx, chunk, pickle.dumps(tokens))
            )
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error storing file in database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_all_documents_from_db():
    """Retrieve all documents and their tokenized versions from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all chunks with file information
    cursor.execute('''
    SELECT f.filename, c.chunk_index, c.chunk_text, c.tokens 
    FROM chunks c 
    JOIN files f ON c.file_id = f.id
    ORDER BY f.filename, c.chunk_index
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return [], [], []
    
    file_names = []
    documents = []
    tokenized_docs = []
    
    for filename, chunk_idx, chunk_text, tokens_blob in results:
        display_name = f"{filename} (Part {chunk_idx+1})"
        file_names.append(display_name)
        documents.append(chunk_text)
        tokenized_docs.append(pickle.loads(tokens_blob))
    
    return documents, tokenized_docs, file_names

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stem tokens
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def split_into_chunks(text, chunk_size=1000):
    """Split text into chunks of approximately equal size"""
    # Split by sentences first to avoid cutting in the middle of a sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= chunk_size or not current_chunk:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def build_bm25_model(documents, tokenized_docs=None):
    """Build BM25 model from preprocessed documents"""
    if tokenized_docs is None:
        tokenized_docs = [preprocess_text(doc) for doc in documents]
    return BM25Okapi(tokenized_docs), tokenized_docs

def calculate_lcs_length(X, Y):
    """Calculate longest common subsequence length between two token lists"""
    m, n = len(X), len(Y)
    
    # Create a table to store lengths of LCS for all sub-problems
    L = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Build L[m+1][n+1] in bottom-up fashion
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    # L[m][n] contains the length of LCS
    return L[m][n]

def calculate_rouge_l(reference_tokens, prediction_tokens):
    """Calculate ROUGE-L score between reference and prediction tokens"""
    lcs_length = calculate_lcs_length(reference_tokens, prediction_tokens)
    
    if len(prediction_tokens) == 0:
        precision = 0.0
    else:
        precision = lcs_length / len(prediction_tokens)
        
    if len(reference_tokens) == 0:
        recall = 0.0
    else:
        recall = lcs_length / len(reference_tokens)
    
    if precision + recall == 0:
        f_measure = 0.0
    else:
        f_measure = (2 * precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure,
        'lcs_length': lcs_length
    }

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def calculate_meteor(reference_tokens, prediction_tokens):
    """Calculate METEOR score between reference and prediction tokens"""
    # Basic METEOR (exact matches)
    reference_counter = Counter(reference_tokens)
    prediction_counter = Counter(prediction_tokens)
    
    matches = sum((reference_counter & prediction_counter).values())
    basic_precision = matches / len(prediction_tokens) if prediction_tokens else 0
    basic_recall = matches / len(reference_tokens) if reference_tokens else 0
    
    if basic_precision + basic_recall == 0:
        basic_f_score = 0
    else:
        basic_f_score = (2 * basic_precision * basic_recall) / (basic_precision + basic_recall)
    
    # METEOR with synonym expansion
    synonym_matches = 0
    for token in prediction_tokens:
        if token in reference_tokens:
            continue  # Already counted in exact matches
        
        token_synonyms = get_synonyms(token)
        for ref_token in reference_tokens:
            if ref_token in token_synonyms:
                synonym_matches += 1
                break
    
    total_matches = matches + synonym_matches
    syn_precision = total_matches / len(prediction_tokens) if prediction_tokens else 0
    syn_recall = total_matches / len(reference_tokens) if reference_tokens else 0
    
    if syn_precision + syn_recall == 0:
        meteor_score = 0
    else:
        meteor_score = (2 * syn_precision * syn_recall) / (syn_precision + syn_recall)
    
    return {
        'basic_meteor': basic_f_score,
        'meteor_with_synonyms': meteor_score,
        'exact_matches': matches,
        'synonym_matches': synonym_matches
    }

def search_documents(query, bm25_model, tokenized_documents, original_docs, file_names, top_n=5):
    """Search documents using BM25"""
    # Preprocess the query
    tokenized_query = preprocess_text(query)
    
    # Get scores for each document
    doc_scores = bm25_model.get_scores(tokenized_query)
    
    # Get top N documents
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
    
    results = []
    for idx in top_indices:
        if doc_scores[idx] > 0:  # Only include documents with positive scores
            results.append({
                'document': original_docs[idx],
                'score': doc_scores[idx],
                'tokens': tokenized_documents[idx],
                'filename': file_names[idx] if file_names else f"Document {idx+1}",
                'query_tokens': tokenized_query  # Store query tokens for evaluation
            })
    
    return results

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and store in database"""
    documents = []
    file_names = []
    tokenized_docs = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text:
            # Split large documents into chunks
            chunks = split_into_chunks(text)
            
            # Tokenize chunks
            tokens_list = [preprocess_text(chunk) for chunk in chunks]
            
            # Store in database
            store_file_in_db(file, chunks, tokens_list)
            
            # Add to current session data
            documents.extend(chunks)
            file_names.extend([f"{file.name} (Part {i+1})" for i in range(len(chunks))])
            tokenized_docs.extend(tokens_list)
    
    return documents, tokenized_docs, file_names

# Streamlit UI
def main():
    st.title("PDF Search Engine with BM25")
    
    # Initialize database if it doesn't exist
    init_database()
    
    # Check if database has content
    has_content = has_database_content()
    
    # Session state initialization
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        st.session_state.tokenized_docs = []
        st.session_state.file_names = []
        st.session_state.bm25_model = None
    
    # Load existing data if available
    if has_content and not st.session_state.documents:
        with st.spinner("Loading existing documents from database..."):
            documents, tokenized_docs, file_names = get_all_documents_from_db()
            if documents:
                st.session_state.documents = documents
                st.session_state.tokenized_docs = tokenized_docs
                st.session_state.file_names = file_names
                st.session_state.bm25_model, _ = build_bm25_model(
                    documents, tokenized_docs
                )
                st.success(f"Loaded {len(documents)} document chunks from database")
    
    # File upload section
    st.subheader("Upload New PDF Files")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("Process Uploaded Files")
        if process_button:
            with st.spinner("Processing PDFs..."):
                new_docs, new_tokens, new_names = process_uploaded_files(uploaded_files)
                
                # Update session data
                if new_docs:
                    st.session_state.documents.extend(new_docs)
                    st.session_state.tokenized_docs.extend(new_tokens)
                    st.session_state.file_names.extend(new_names)
                    st.session_state.bm25_model, _ = build_bm25_model(
                        st.session_state.documents, st.session_state.tokenized_docs
                    )
                    st.success(f"Added {len(new_docs)} new document chunks to the database")
    
    # Show database status
    if st.session_state.documents:
        st.info(f"Database contains {len(st.session_state.documents)} document chunks from {len(set([name.split(' (Part')[0] for name in st.session_state.file_names]))} files")
    else:
        st.info("No documents in database. Please upload PDF files to begin.")
        return
    
    # Search interface
    st.subheader("Search Documents")
    query = st.text_input("Enter your search query")
    
    if query and st.session_state.bm25_model:
        with st.spinner("Searching..."):
            results = search_documents(
                query, 
                st.session_state.bm25_model, 
                st.session_state.tokenized_docs, 
                st.session_state.documents,
                st.session_state.file_names
            )
        
        if results:
            st.subheader("Search Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result['filename']} (Score: {result['score']:.2f})"):
                    # Highlight matching terms in the document
                    highlighted_text = result['document']
                    for term in preprocess_text(query):
                        pattern = r'\b' + re.escape(term) + r'[a-zA-Z]*\b'
                        highlighted_text = re.sub(
                            pattern, 
                            lambda m: f"**{m.group(0)}**", 
                            highlighted_text, 
                            flags=re.IGNORECASE
                        )
                    
                    st.markdown(highlighted_text)
                    
                    # Use checkbox to toggle details
                    if st.checkbox(f"Show document details for result {i}", key=f"details_{i}"):
                        st.markdown("**Document Details:**")
                        st.write(f"Document length: {len(result['document'])} characters")
                        st.write(f"Tokens after preprocessing: {', '.join(result['tokens'][:25])}...")
                        
                        # Calculate and display ROUGE-L scores
                        rouge_scores = calculate_rouge_l(result['query_tokens'], result['tokens'])
                        
                        st.markdown("### ROUGE-L Calculation Details:")
                        st.write(f"Token reference sentence (after stemming): {result['query_tokens']} ... ({len(result['query_tokens'])} tokens in total)")
                        st.write(f"Prediction sentence tokens (after stemming): {result['tokens'][:10]} ... (total {len(result['tokens'])} tokens)")
                        st.write(f"Number of reference sentence tokens: {len(result['query_tokens'])}")
                        st.write(f"Number of prediction sentence tokens: {len(result['tokens'])}")
                        st.write(f"LCS length: {rouge_scores['lcs_length']}")
                        st.write(f"Precision = LCS / Number of prediction tokens = {rouge_scores['lcs_length']} / {len(result['tokens'])} = {rouge_scores['precision']:.4f} = {rouge_scores['precision']*100:.0f}%")
                        st.write(f"Recall = LCS / Number of reference tokens = {rouge_scores['lcs_length']} / {len(result['query_tokens'])} = {rouge_scores['recall']:.4f} = {rouge_scores['recall']*100:.0f}%")
                        st.write(f"F-measure = (2 * Precision * Recall) / (Precision + Recall)")
                        st.write(f"= (2 * {rouge_scores['precision']:.4f} * {rouge_scores['recall']:.4f}) / ({rouge_scores['precision']:.4f} + {rouge_scores['recall']:.4f})")
                        st.write(f"= {rouge_scores['f_measure']:.4f} = {rouge_scores['f_measure']*100:.0f}%")
                        
                        # Calculate and display METEOR scores
                        meteor_scores = calculate_meteor(result['query_tokens'], result['tokens'])
                        
                        st.markdown("### METEOR Calculation Details:")
                        st.write(f"METEOR Score: {meteor_scores['meteor_with_synonyms']*100:.0f}%")
                        st.write("METEOR is calculated based on matching tokens between reference and prediction sentences with synonym expansion.")
                        st.write(f"Basic METEOR (without synonym expansion): {meteor_scores['basic_meteor']*100:.0f}%")
                        st.write(f"METEOR with synonym expansion: {meteor_scores['meteor_with_synonyms']*100:.0f}%")
                        st.write(f"Exact matches: {meteor_scores['exact_matches']}")
                        st.write(f"Synonym matches: {meteor_scores['synonym_matches']}")
        else:
            st.info("No matching documents found")

if __name__ == "__main__":
    main()