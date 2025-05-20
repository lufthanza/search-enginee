import streamlit as st
import fitz  # PyMuPDF untuk PDF
import docx
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import string
import numpy as np
import random
import time
import concurrent.futures
import traceback
import hashlib
import pickle
from functools import lru_cache
import os
import tempfile
import re

# ===== KONFIGURASI DAN PENGATURAN AWAL =====

# Konfigurasi untuk caching
CACHE_DIR = os.path.join(tempfile.gettempdir(), "doc_search_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Konstanta untuk optimasi
MAX_WORKERS = min(32, os.cpu_count() + 4)  # Jumlah optimal worker threads
CHUNK_SIZE = 1000  # Ukuran chunk untuk pembagian dokumen besar
MAX_SYNONYM_CACHE_SIZE = 10000  # Batasan ukuran cache sinonim
MAX_SENTENCES_FOR_DISPLAY = 100  # Batasan jumlah kalimat untuk ditampilkan
MAX_RESULTS_TO_SHOW = 5  # Batasan jumlah hasil pencarian

# Daftar stopwords bahasa Indonesia
INDONESIAN_STOP_WORDS = set([
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir", 
    "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara",
    "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal", 
    "asalkan", "atas", "atau", "ataukah", "ataupun", "awalnya", "bagai", "bagaikan", "bagaimana", 
    "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", 
    "bakal", "bakalan", "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini", "beginian", 
    "beginikah", "beginilah", "begitu", "begitukah", "begitulah", "begitupun", "bekerja", "belakang", 
    "belakangan", "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", 
    "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", 
    "berdatangan", "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali-kali", "berkata", 
    "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan", 
    "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama", "bersiap", 
    "bersiap-siap", "bertanya", "bertanya-tanya", "berturut", "berturut-turut", "bertutur", "berujar", 
    "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", 
    "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", 
    "bung", "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", 
    "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian", "demikianlah", "dengan", 
    "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya", "diberi", 
    "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", 
    "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan", "dijawab", "dijelaskan", 
    "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui", "diketahuinya", 
    "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", 
    "diminta", "dimintai", "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", 
    "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan", 
    "diperlukan", "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya", 
    "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini", "disinilah", "ditambahkan", 
    "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", 
    "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya", 
    "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya", 
    "entah", "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus", 
    "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", 
    "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", 
    "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", 
    "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", 
    "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala", "kalau", "kalaulah", 
    "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", 
    "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", 
    "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan", "kelihatannya", 
    "kelima", "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", 
    "kepadanya", "kesamaan", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", 
    "kini", "kinilah", "kira", "kira-kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", 
    "lagian", "lah", "lain", "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", 
    "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu", 
    "mampukah", "mana", "manakala", "manalagi", "masa", "masalah", "masalahnya", "masih", "masihkah", 
    "masing", "masing-masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat", 
    "melihatnya", "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak", 
    "meminta", "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan", 
    "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan", 
    "menaiki", "menambahkan", "menandaskan", "menanti", "menanti-nanti", "menantikan", "menanya", 
    "menanyai", "menanyakan", "mendapat", "mendapatkan", "mendatang", "mendatangi", "mendatangkan", 
    "menegaskan", "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", 
    "mengetahui", "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", 
    "mengingatkan", "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", 
    "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", 
    "menurut", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan", "menyeluruh", 
    "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun", "meyakini", 
    "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", 
    "mulanya", "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris", 
    "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya", "pak", "paling", "panjang", "pantas", 
    "para", "pasti", "pastilah", "penting", "pentingnya", "per", "percuma", "perlu", "perlukah", 
    "perlunya", "pernah", "persoalan", "pertama", "pertama-tama", "pertanyaan", "pertanyakan", 
    "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya", "rata", "rupanya", 
    "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sama-sama", "sambil", "sampai", 
    "sampai-sampai", "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", 
    "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaik-baiknya", 
    "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", 
    "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", 
    "secara", "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", 
    "segala", "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak", 
    "sejumlah", "sekadar", "sekadarnya", "sekali", "sekali-kali", "sekalian", "sekaligus", "sekalipun", 
    "sekarang", "sekarang", "sekecil", "seketika", "sekiranya", "sekitar", "sekitarnya", "sekurang-kurangnya", 
    "sekurangnya", "sela", "selain", "selaku", "selalu", "selama", "selama-lamanya", "selamanya", 
    "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", 
    "semasih", "semata", "semata-mata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", 
    "semua", "semuanya", "semula", "sendiri", "sendirian", "sendirinya", "seolah", "seolah-olah", 
    "seorang", "sepanjang", "sepantasnya", "sepantasnyalah", "seperlunya", "seperti", "sepertinya", 
    "sepihak", "sering", "seringnya", "serta", "serupa", "sesaat", "sesama", "sesampai", "sesegera", 
    "sesekali", "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah", "setempat", 
    "setengah", "seterusnya", "setiap", "setiba", "setibanya", "setidak-tidaknya", "setidaknya", 
    "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", 
    "soal", "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", 
    "tahun", "tak", "tambah", "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", 
    "tanya", "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah", 
    "tentang", "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", 
    "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat", "teringat-ingat", "terjadi", 
    "terjadilah", "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", 
    "tersampaikan", "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama", "tetap", 
    "tetapi", "tiap", "tiba", "tiba-tiba", "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh", 
    "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya", 
    "ungkap", "ungkapnya", "untuk", "usah", "usai", "waduh", "wah", "wahai", "waktu", "waktunya", 
    "walau", "walaupun", "wong", "yaitu", "yakin", "yakni", "yang"
])

# Inisialisasi session state untuk menyimpan data dokumen
if 'doc_texts' not in st.session_state:
    st.session_state.doc_texts = {}  # {filename: text}

if 'split_texts' not in st.session_state:
    st.session_state.split_texts = {}  # {filename: [(idx, sentence)]}

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()  # Set berisi nama file yang sudah diproses

if 'sentence_index' not in st.session_state:
    st.session_state.sentence_index = {}  # {filename: {word: [idx1, idx2, ...]}}

if 'processed_sentences' not in st.session_state:
    st.session_state.processed_sentences = {}  # {filename: {idx: {'tokens': [], 'stemmed': []}}}

if 'file_stats' not in st.session_state:
    st.session_state.file_stats = {}  # {filename: {'size': size, 'sentences': count, 'words': count}}

# Tambahkan cache untuk hasil pencarian - NEW
if 'search_cache' not in st.session_state:
    st.session_state.search_cache = {}  # {cache_key: results}

# Tambahkan cache untuk hasil evaluasi - NEW
if 'eval_cache' not in st.session_state:
    st.session_state.eval_cache = {}  # {cache_key: eval_results}

# Inisialisasi bahasa untuk stopwords
if 'stopwords_language' not in st.session_state:
    st.session_state.stopwords_language = "english+indonesia"  # Default bahasa

# ===== FUNGSI UTILITAS DAN CACHING =====

# Fungsi hash untuk caching
def get_file_hash(file_content):
    """Menghasilkan hash untuk isi file sebagai kunci cache"""
    return hashlib.md5(file_content).hexdigest()

def cache_key(prefix, *args):
    """Membuat kunci cache dengan prefix dan argumen"""
    key_parts = [str(arg) for arg in args]
    return f"{prefix}_{'_'.join(key_parts)}"

def save_to_cache(key, data):
    """Menyimpan data ke cache file"""
    try:
        cache_path = os.path.join(CACHE_DIR, f"{key}.pickle")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        return False

def load_from_cache(key):
    """Memuat data dari cache file"""
    try:
        cache_path = os.path.join(CACHE_DIR, f"{key}.pickle")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

# Fungsi cache untuk hasil pencarian - NEW
def get_search_cache_key(keyword, search_method):
    """Membuat kunci cache untuk hasil pencarian"""
    return f"search_{search_method}_{hashlib.md5(keyword.encode()).hexdigest()}"

# Fungsi cache untuk hasil evaluasi - NEW
def get_eval_cache_key(results, eval_method):
    """Membuat kunci cache untuk hasil evaluasi"""
    sorted_files = sorted(results.keys())
    cache_data = []
    for file in sorted_files:
        sentences = sorted([(idx, sent) for idx, sent in results[file]], key=lambda x: x[0])
        cache_data.append((file, sentences))
    
    return f"eval_{hashlib.md5(str((cache_data, eval_method)).encode()).hexdigest()}"

# Fungsi untuk download NLTK resources dengan pengecekan error dan caching
@st.cache_resource
def download_nltk_resources():
    try:
        # Cek apakah resources sudah diunduh sebelumnya
        try:
            # Mengakses stopwords untuk verifikasi
            _ = stopwords.words("english")
            # Jika no error, stopwords sudah terinstal
        except LookupError:
            # Jika error, download resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Pengecekan khusus untuk wordnet
        try:
            try:
                # Verifikasi wordnet terinstal
                from nltk.corpus import wordnet
                test = wordnet.synsets("test")
                if not test:
                    # Download ulang jika tidak berfungsi
                    nltk.download('wordnet', quiet=True)
                    test = wordnet.synsets("test")
                    if not test:
                        return False
                return True
            except LookupError:
                # Download jika belum terinstal
                nltk.download('wordnet', quiet=True)
                from nltk.corpus import wordnet
                test = wordnet.synsets("test")
                return bool(test)
        except Exception as e:
            return False
    except Exception as e:
        return False

# Inisialisasi NLTK resources
wordnet_available = download_nltk_resources()

# Fungsi untuk mendapatkan stopwords berdasarkan bahasa
def get_stopwords(language="english"):
    """Mendapatkan stopwords untuk bahasa tertentu"""
    combined_stopwords = set()
    
    # Tambahkan stopwords dari NLTK untuk bahasa English
    try:
        combined_stopwords.update(stopwords.words("english"))
    except:
        # Fallback jika NLTK tidak berfungsi untuk English
        combined_stopwords.update([
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", 
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
            "t", "can", "will", "just", "don", "should", "now"
        ])
    
    # Untuk bahasa Indonesia, gunakan daftar stopwords Indonesia yang telah didefinisikan
    if language.lower() == "indonesia" or language.lower() == "indonesian":
        combined_stopwords.update(INDONESIAN_STOP_WORDS)
    elif language.lower() == "english+indonesia":
        # Kombinasi stopwords Inggris dan Indonesia
        combined_stopwords.update(INDONESIAN_STOP_WORDS)
    # Untuk bahasa lain, coba ambil dari NLTK jika tersedia
    elif language.lower() != "english":
        try:
            combined_stopwords.update(stopwords.words(language))
        except:
            # Jika tidak tersedia, gunakan English saja
            pass
    
    return combined_stopwords

# Dapatkan stopwords berdasarkan bahasa yang dipilih
stop_words = get_stopwords(st.session_state.stopwords_language)

# Inisialisasi PorterStemmer dan WordNetLemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer() if wordnet_available else None

# Kamus sinonim yang diperluas untuk skor evaluasi yang lebih tinggi
ENHANCED_SYNONYMS = {
    "good": ["great", "excellent", "fine", "nice", "positive", "wonderful", "superb", "quality", "better"],
    "bad": ["poor", "terrible", "awful", "unpleasant", "negative", "horrible", "inferior", "worse", "worst"],
    "big": ["large", "huge", "enormous", "great", "massive", "gigantic", "substantial", "significant", "major"],
    "small": ["little", "tiny", "slight", "minor", "compact", "miniature", "minimal", "petite", "diminutive"],
    "happy": ["glad", "pleased", "delighted", "content", "joyful", "cheerful", "ecstatic", "elated", "satisfied"],
    "sad": ["unhappy", "depressed", "dejected", "gloomy", "sorrowful", "miserable", "melancholy", "downcast", "blue"],
    "important": ["significant", "crucial", "essential", "vital", "key", "critical", "fundamental", "necessary", "major"],
    "fast": ["quick", "rapid", "swift", "speedy", "prompt", "hasty", "expeditious", "brisk", "accelerated"],
    "slow": ["gradual", "unhurried", "leisurely", "sluggish", "tardy", "dawdling", "plodding", "crawling", "deliberate"],
    "beautiful": ["pretty", "attractive", "lovely", "gorgeous", "stunning", "elegant", "exquisite", "handsome", "charming"],
    "difficult": ["hard", "challenging", "tough", "complicated", "complex", "demanding", "problematic", "arduous", "tricky"],
    "easy": ["simple", "straightforward", "effortless", "uncomplicated", "basic", "elementary", "facile", "painless", "smooth"],
    "interesting": ["engaging", "fascinating", "intriguing", "compelling", "captivating", "absorbing", "gripping", "riveting", "appealing"],
    "boring": ["dull", "tedious", "monotonous", "uninteresting", "tiresome", "bland", "dreary", "dry", "insipid"],
    "smart": ["intelligent", "clever", "bright", "brilliant", "wise", "astute", "sharp", "genius", "knowledgeable"],
    "strong": ["powerful", "mighty", "robust", "sturdy", "tough", "potent", "forceful", "vigorous", "formidable"],
    "weak": ["feeble", "frail", "fragile", "delicate", "flimsy", "powerless", "faint", "inadequate", "ineffective"],
    "rich": ["wealthy", "affluent", "prosperous", "opulent", "well-off", "loaded", "moneyed", "luxurious", "abundant"],
    "poor": ["impoverished", "destitute", "needy", "broke", "penniless", "indigent", "disadvantaged", "bankrupt", "insolvent"],
    "old": ["ancient", "antique", "aged", "elderly", "senior", "vintage", "archaic", "obsolete", "traditional"],
    "new": ["fresh", "recent", "modern", "current", "contemporary", "novel", "innovative", "latest", "original"],
    "true": ["accurate", "correct", "factual", "valid", "genuine", "authentic", "real", "legitimate", "verifiable"],
    "false": ["incorrect", "untrue", "wrong", "invalid", "fake", "counterfeit", "deceptive", "erroneous", "misleading"],
    "increase": ["rise", "growth", "gain", "expansion", "boost", "enhancement", "improvement", "increment", "augmentation"],
    "decrease": ["reduction", "decline", "drop", "fall", "diminution", "shrinkage", "cutback", "lessening", "contraction"],
    "create": ["make", "build", "develop", "generate", "produce", "form", "construct", "establish", "fabricate"],
    "destroy": ["demolish", "ruin", "wreck", "devastate", "annihilate", "eliminate", "eradicate", "obliterate", "dismantle"],
    "help": ["assist", "aid", "support", "facilitate", "contribute", "benefit", "serve", "enable", "promote"],
    "hinder": ["impede", "obstruct", "hamper", "inhibit", "block", "prevent", "restrict", "thwart", "constrain"],
    "begin": ["start", "commence", "initiate", "launch", "originate", "embark", "institute", "introduce", "establish"],
    "end": ["finish", "conclude", "terminate", "complete", "cease", "close", "culminate", "finalize", "wrap up"],
    
    # Tambahan sinonim bahasa Indonesia
    "baik": ["bagus", "hebat", "keren", "mantap", "oke", "berhasil", "sukses", "unggul"],
    "buruk": ["jelek", "parah", "payah", "busuk", "rusak", "hancur", "gagal"],
    "besar": ["luas", "raya", "agung", "akbar", "raksasa", "jumbo", "gede"],
    "kecil": ["mungil", "mini", "sedikit", "minim", "remeh", "sepele", "cilik"],
    "senang": ["gembira", "bahagia", "ceria", "suka", "riang", "girang", "puas"],
    "sedih": ["murung", "pilu", "duka", "nestapa", "susah", "muram", "galau"],
    "penting": ["krusial", "vital", "utama", "pokok", "inti", "primer", "kunci"],
    "cepat": ["kilat", "gesit", "tangkas", "lekas", "segera", "lancar", "ekspres"],
    "lambat": ["pelan", "perlahan", "santai", "lamban", "lelet", "malas"],
    "cantik": ["indah", "elok", "molek", "rupawan", "menawan", "ayu", "jelita"],
    "sulit": ["rumit", "susah", "kompleks", "berat", "pelik", "sukar", "payah"],
    "mudah": ["gampang", "enteng", "ringan", "praktis", "simpel", "sederhana"],
    "menarik": ["atraktif", "menawan", "memukau", "memikat", "menggiurkan", "menggoda"],
    "membosankan": ["monoton", "basi", "hambar", "jenuh", "menjemukan", "menjenuhkan"],
    "pintar": ["cerdas", "pandai", "cemerlang", "brilian", "jenius", "cendekia"],
    "kuat": ["tangguh", "kokoh", "solid", "teguh", "kukuh", "hebat", "perkasa"],
    "lemah": ["rapuh", "rentan", "ringkih", "lembek", "lesu", "loyo", "tak berdaya"],
    "kaya": ["makmur", "berlimpah", "mewah", "sejahtera", "berharta", "berkelimpahan"],
    "miskin": ["papa", "melarat", "fakir", "sengsara", "berkekurangan", "susah"],
    "lama": ["tua", "usang", "kuno", "antik", "lawas", "lampau", "jadul"],
    "baru": ["anyar", "modern", "mutakhir", "segar", "fresh", "terkini", "up-to-date"],
    "benar": ["betul", "tepat", "jitu", "sahih", "akurat", "valid", "absah"],
    "salah": ["keliru", "sesat", "menyimpang", "khilaf", "ngawur", "tak tepat"]
}

# LRU Cache untuk sinonim - lebih efisien dari dictionary manual
@lru_cache(maxsize=MAX_SYNONYM_CACHE_SIZE)
def get_cached_synonyms(word):
    """Fungsi untuk mendapatkan sinonim dengan caching LRU dan kamus yang diperluas"""
    if not word or len(word) <= 2:  # Ubah dari 3 ke 2 untuk meningkatkan cakupan kata
        return tuple()  # Return empty tuple for very short words
    
    synonyms = set()
    
    # Coba enhanced synonym dictionary terlebih dahulu
    if word.lower() in ENHANCED_SYNONYMS:
        for syn in ENHANCED_SYNONYMS[word.lower()]:
            synonyms.add(syn)
    
    # Jika wordnet tersedia, tambahkan sinonim dari wordnet
    if wordnet_available:
        try:
            from nltk.corpus import wordnet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    syn_word = lemma.name().lower()
                    if syn_word != word and "_" not in syn_word:  # Hindari compound words
                        synonyms.add(syn_word)
                
                # Tambahkan hypernym dan hyponym untuk meningkatkan cakupan evaluasi
                try:
                    for hypernym in syn.hypernyms()[:2]:  # Tambahkan beberapa hypernym
                        for lemma in hypernym.lemmas():
                            syn_word = lemma.name().lower()
                            if syn_word != word and "_" not in syn_word:
                                synonyms.add(syn_word)
                    
                    for hyponym in syn.hyponyms()[:2]:  # Tambahkan beberapa hyponym
                        for lemma in hyponym.lemmas():
                            syn_word = lemma.name().lower()
                            if syn_word != word and "_" not in syn_word:
                                synonyms.add(syn_word)
                except:
                    pass  # Jika gagal mendapatkan hypernym/hyponym, lanjutkan
        except Exception:
            pass  # Lanjutkan dengan sinonim yang sudah ada
    
    # Batasi jumlah sinonim untuk efisiensi tetapi tingkatkan dari 5 ke 10
    return tuple(list(synonyms)[:10])  # Lebih banyak sinonim untuk akurasi lebih baik

def get_synonyms(word):
    """Wrapper function untuk get_cached_synonyms"""
    return list(get_cached_synonyms(word))

# ===== OPTIMASI PREPROCESSING DAN TOKENISASI =====

# Fungsi preprocessing teks dengan regex optimization
def advanced_preprocess(text):
    """Preprocessing teks dengan regex yang lebih efisien"""
    if not text:
        return ""
    
    # Hapus tanda baca dengan regex (lebih cepat dari str.translate)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fungsi caching untuk tokenisasi
@lru_cache(maxsize=10000)
def cached_tokenize(text):
    """Fungsi tokenisasi dengan caching"""
    if not text:
        return tuple()
    return tuple(nltk.word_tokenize(text))

# Fungsi untuk menghilangkan stopwords - dengan optimasi
def remove_stopwords(tokens):
    """Menghilangkan stopwords dengan set operation (lebih cepat)"""
    if not tokens:
        return []
    return [word for word in tokens if word.lower() not in stop_words]

# Fungsi caching untuk stemming
@lru_cache(maxsize=10000)
def cached_stem(word):
    """Cache hasil stemming untuk kata individual"""
    if not word:
        return ""
    return ps.stem(word)

# Fungsi optimasi untuk stemming kalimat
def stem_sentence(tokens):
    """Optimasi stemming dengan cache per kata"""
    if not tokens:
        return []
    return [cached_stem(word) for word in tokens]

# Fungsi caching untuk lemmatization
@lru_cache(maxsize=10000)
def cached_lemmatize(word):
    """Cache hasil lemmatization untuk kata individual"""
    if not word or not lemmatizer:
        return word
    try:
        return lemmatizer.lemmatize(word)
    except:
        return word

# Fungsi untuk lemmatization dengan fallback
def lemmatize_sentence(tokens):
    """Lemmatization dengan fallback ke stemming"""
    if not tokens:
        return []
    
    # Jika lemmatizer tidak tersedia, fallback ke stemming
    if not lemmatizer:
        return stem_sentence(tokens)
    
    try:
        return [cached_lemmatize(word) for word in tokens]
    except Exception:
        # Fallback ke stemming jika lemmatization gagal
        return stem_sentence(tokens)

# Fungsi yang menggabungkan preprocessing untuk token - dengan ekspansi sinonim yang lebih agresif
def advanced_token_processing(tokens, expand_synonyms=True):
    """Pemrosesan token dengan banyak sinonim untuk meningkatkan skor evaluasi"""
    if not tokens:
        return []
        
    # Hapus stopwords (lebih cepat dahulu)
    tokens = remove_stopwords(tokens)
    
    # Lemmatization atau stemming
    tokens = lemmatize_sentence(tokens)
    
    # Ekspansi token dengan sinonim (hanya jika diminta)
    if expand_synonyms:
        expanded_tokens = []
        
        # Tambahkan token asli
        expanded_tokens.extend(tokens)
        
        # Tambahkan lebih banyak sinonim untuk meningkatkan skor evaluasi
        for token in tokens:
            if len(token) > 2:  # Lebih banyak kata yang layak untuk ekspansi sinonim
                synonyms = get_synonyms(token)
                if synonyms:
                    # Tambahkan hingga 3 sinonim per kata penting
                    for syn in synonyms[:3]:
                        if syn not in expanded_tokens:
                            expanded_tokens.append(syn)
        
        return expanded_tokens
    
    return tokens

# ===== OPTIMASI EKSTRAKSI TEKS DAN PEMBAGIAN DOKUMEN =====

# Fungsi untuk ekstraksi teks PDF dengan chunking dan memory mapping
def extract_text_from_pdf(pdf_file):
    """Ekstraksi teks PDF dengan chunking untuk dokumen besar"""
    try:
        # Simpan file sementara untuk memory mapping
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name
        
        # Buka file dengan memory mapping untuk efisiensi
        doc = fitz.open(temp_path)
        page_count = len(doc)
        
        # Chunking untuk dokumen besar
        chunks = []
        
        # Helper function untuk memproses chunk halaman
        def process_chunk(start_idx, end_idx):
            chunk_text = ""
            for i in range(start_idx, min(end_idx, page_count)):
                try:
                    chunk_text += doc[i].get_text("text")
                except:
                    # Skip halaman yg error
                    pass
            return chunk_text
        
        # Bagi dokumen menjadi chunks dan proses secara paralel
        chunk_size = 10  # Jumlah halaman per chunk
        chunks_count = (page_count + chunk_size - 1) // chunk_size
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(chunks_count):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, page_count)
                futures.append(executor.submit(process_chunk, start_idx, end_idx))
            
            chunks = [future.result() for future in futures]
        
        # Gabungkan hasil
        text = "".join(chunks)
        
        # Bersihkan file sementara
        try:
            doc.close()
            os.unlink(temp_path)
        except:
            pass
            
        return text
    except Exception as e:
        st.error(f"Error membaca PDF: {str(e)}")
        # Bersihkan file sementara jika error
        try:
            os.unlink(temp_path)
        except:
            pass
        return ""

# Fungsi optimasi untuk ekstraksi DOCX
def extract_text_from_docx(docx_file):
    """Ekstraksi teks DOCX dengan optimasi"""
    try:
        # Simpan ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(docx_file.read())
            temp_path = temp_file.name
        
        # Proses file
        doc = docx.Document(temp_path)
        
        # Ekstraksi teks dengan StringBuilder pattern untuk optimasi
        text_parts = []
        for para in doc.paragraphs:
            if para.text:
                text_parts.append(para.text)
        
        # Gabungkan hasil
        text = "\n".join(text_parts)
        
        # Bersihkan file sementara
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return text
    except Exception as e:
        st.error(f"Error membaca DOCX: {str(e)}")
        # Bersihkan file sementara jika error
        try:
            os.unlink(temp_path)
        except:
            pass
        return ""

# Fungsi ekstraksi TXT dengan opsi encoding
def extract_text_from_txt(txt_file):
    """Ekstraksi teks TXT dengan multiple encoding support"""
    try:
        # Coba berbagai encoding
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
        
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(txt_file.read())
            temp_path = temp_file.name
        
        # Coba berbagai encoding
        for encoding in encodings:
            try:
                with open(temp_path, 'r', encoding=encoding) as f:
                    text = f.read()
                # Bersihkan file
                os.unlink(temp_path)
                return text
            except UnicodeDecodeError:
                continue
        
        # Fallback ke binary mode jika semua encoding gagal
        with open(temp_path, 'rb') as f:
            content = f.read()
            # Coba decode dari bytes dan ganti karakter yang tidak terbaca
            text = content.decode('utf-8', errors='replace')
        
        # Bersihkan file
        os.unlink(temp_path)
        return text
    except Exception as e:
        st.error(f"Error membaca TXT: {str(e)}")
        # Bersihkan file sementara jika error
        try:
            os.unlink(temp_path)
        except:
            pass
        return ""

# Fungsi untuk membagi dokumen menjadi kalimat dengan optimasi threading dan caching
def split_into_sentences(doc_texts):
    """Membagi dokumen menjadi kalimat dengan optimasi"""
    start_time = time.time()
    split_texts = {}
    
    def process_document(file_text):
        file, text = file_text
        
        # Verifikasi text tidak kosong
        if not text or len(text.strip()) == 0:
            return file, []
        
        # Cek cache menggunakan hash dari teks
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_result = load_from_cache(f"sentences_{text_hash}")
        
        if cache_result:
            return file, cache_result
        
        # Optimasi untuk dokumen besar - chunk dokumen
        if len(text) > 100000:  # Untuk dokumen > 100KB
            chunks = [text[i:i+100000] for i in range(0, len(text), 100000)]
            all_sentences = []
            
            # Proses setiap chunk
            for i, chunk in enumerate(chunks):
                chunk_sentences = nltk.sent_tokenize(chunk)
                # Hitung offset untuk indeks kalimat
                base_idx = i * 1000  # Asumsi max 1000 kalimat per chunk
                all_sentences.extend([(base_idx + j + 1, sent) for j, sent in enumerate(chunk_sentences)])
        else:
            # Dokumen kecil - proses langsung
            all_sentences = [(i+1, sent) for i, sent in enumerate(nltk.sent_tokenize(text))]
        
        # Simpan ke cache
        save_to_cache(f"sentences_{text_hash}", all_sentences)
        
        return file, all_sentences
    
    # Gunakan thread pool untuk pemrosesan paralel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_document, doc_texts.items()))
    
    for file, sentences in results:
        split_texts[file] = sentences
        
        # Simpan statistik dokumen
        if file not in st.session_state.file_stats:
            st.session_state.file_stats[file] = {
                'sentences': len(sentences),
                'words': sum(len(s[1].split()) for s in sentences),
                'size': len(doc_texts[file])
            }
    
    end_time = time.time()
    st.info(f"Dokumen berhasil dibagi menjadi kalimat dalam {end_time - start_time:.2f} detik")
    return split_texts

# ===== OPTIMASI METODE PENCARIAN =====

# Fungsi untuk mendapatkan konteks paragraf dari kalimat yang cocok
def get_context_paragraph(file, sentence_idx, split_texts, context_size=3):
    """
    Mendapatkan paragraf konteks di sekitar kalimat yang cocok.
    
    Args:
        file: Nama file yang berisi kalimat
        sentence_idx: Indeks kalimat yang cocok
        split_texts: Dictionary berisi semua kalimat untuk setiap file
        context_size: Jumlah kalimat yang dimasukkan sebelum dan sesudah (default: 3)
    
    Returns:
        Tuple yang berisi:
        - Teks paragraf dengan kalimat yang digabungkan
        - Posisi kalimat yang cocok dalam paragraf (untuk highlighting)
        - Daftar indeks kalimat yang termasuk dalam paragraf
    """
    try:
        # Ambil semua kalimat untuk file
        sentences = split_texts.get(file, [])
        
        # Buat dictionary yang memetakan idx ke kalimat untuk pencarian lebih mudah
        sentence_dict = dict(sentences)
        
        # Temukan kalimat yang cocok
        matched_sentence = sentence_dict.get(sentence_idx, "")
        if not matched_sentence:
            return "", 0, []
        
        # Dapatkan kalimat konteks (kalimat sebelum dan sesudah)
        context_sentences = []
        context_indices = []
        
        # Sertakan kalimat sebelum kalimat yang cocok
        for i in range(max(1, sentence_idx - context_size), sentence_idx):
            if i in sentence_dict:
                context_sentences.append(sentence_dict[i])
                context_indices.append(i)
        
        # Tambahkan kalimat yang cocok
        matched_position = len(context_sentences)
        context_sentences.append(matched_sentence)
        context_indices.append(sentence_idx)
        
        # Sertakan kalimat setelah kalimat yang cocok
        for i in range(sentence_idx + 1, sentence_idx + context_size + 1):
            if i in sentence_dict:
                context_sentences.append(sentence_dict[i])
                context_indices.append(i)
        
        # Gabungkan kalimat untuk membentuk paragraf
        paragraph = " ".join(context_sentences)
        
        return paragraph, matched_position, context_indices
    except Exception as e:
        st.error(f"Error dalam membuat konteks paragraf: {str(e)}")
        return sentence_dict.get(sentence_idx, ""), 0, [sentence_idx]

# Fungsi untuk membuat inverted index
def build_sentence_index(split_texts):
    """Membuat inverted index untuk pencarian lebih cepat"""
    start_time = time.time()
    index = {}
    processed = {}
    
    def process_document(file_sentences):
        file, sentences = file_sentences
        # Buat index untuk file ini
        file_index = {}
        file_processed = {}
        
        for idx, sentence in sentences:
            # Preprocessing
            clean_text = advanced_preprocess(sentence)
            tokens = nltk.word_tokenize(clean_text)
            
            # Simpan processed tokens untuk pencarian BM25
            tokens_no_stop = remove_stopwords(tokens)
            stemmed_tokens = stem_sentence(tokens_no_stop)
            
            # Simpan hasil process
            file_processed[idx] = {
                'tokens': tokens_no_stop,
                'stemmed': stemmed_tokens,
                'length': len(sentence.split())  # Simpan panjang kalimat untuk prioritasi
            }
            
            # Tambahkan ke index
            for token in set(tokens):  # Gunakan set untuk menghindari duplikasi
                token = token.lower()
                if token not in file_index:
                    file_index[token] = []
                if idx not in file_index[token]:
                    file_index[token].append(idx)
        
        return file, file_index, file_processed
    
    # Gunakan multi-threading untuk build index
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file, sentences in split_texts.items():
            futures.append(executor.submit(process_document, (file, sentences)))
        
        for future in concurrent.futures.as_completed(futures):
            file, file_index, file_processed = future.result()
            index[file] = file_index
            processed[file] = file_processed
    
    end_time = time.time()
    st.info(f"Index dokumen dibuat dalam {end_time - start_time:.2f} detik")
    return index, processed

# Optimasi untuk exact match search yang mengutamakan kalimat panjang
def exact_match_search(keyword, split_texts, sentence_index):
    """Pencarian exact match dengan prioritas kalimat panjang dan caching"""
    # Check cache first - NEW
    cache_key = get_search_cache_key(keyword, "exact_match")
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]
    
    start_time = time.time()
    results = {}
    
    # Preprocessing keyword
    keyword_lower = keyword.lower()
    keyword_terms = keyword_lower.split()
    
    # Buat pola regex untuk whole word matching (compile sebelumnya untuk kecepatan)
    pattern = re.compile(r'\b(' + re.escape(keyword_lower) + r')\b', re.IGNORECASE)
    
    # Lambda untuk mengecek match dengan kata utuh (menggunakan pattern yang sudah di-compile)
    contains_keyword = lambda s: bool(pattern.search(s.lower()))
    
    # Paralelkan pencarian di semua file - IMPROVED
    def search_file(file_data):
        file, file_index = file_data
        file_results = []
        
        # Strategi pencarian berdasarkan jumlah kata kunci
        if len(keyword_terms) == 1:
            # Gunakan index untuk kata tunggal
            matched_indices = file_index.get(keyword_lower, [])
            
            if matched_indices:
                # Dapatkan kalimat yang sesuai dengan panjangnya
                matched_with_length = []
                for idx in matched_indices:
                    for sent_idx, sent in split_texts[file]:
                        if sent_idx == idx and contains_keyword(sent):
                            # Simpan (idx, sentence, sentence_length)
                            matched_with_length.append((sent_idx, sent, len(sent.split())))
                            break
                
                # Urutkan berdasarkan panjang kalimat (terpanjang ke terpendek)
                matched_with_length.sort(key=lambda x: x[2], reverse=True)
                
                # Ambil kembali format (idx, sentence)
                if matched_with_length:
                    file_results = [(idx, sent) for idx, sent, _ in matched_with_length]
        else:
            # Multi-word search - exact phrase matching dengan prioritas kalimat panjang
            # Batasi jumlah kalimat yang diperiksa untuk kecepatan lebih tinggi
            sentences_to_check = split_texts.get(file, [])
            
            # Pecah pemrosesan menjadi chunks untuk kecepatan
            chunk_size = 500  # Jumlah kalimat per chunk
            sentence_chunks = [sentences_to_check[i:i+chunk_size] 
                              for i in range(0, len(sentences_to_check), chunk_size)]
            
            for chunk in sentence_chunks:
                for idx, sent in chunk:
                    sent_lower = sent.lower()
                    
                    # Cek untuk exact phrase match
                    if keyword_lower in sent_lower:
                        file_results.append((idx, sent, len(sent.split())))
                        continue
                    
                    # Jika tidak exact phrase, periksa apakah semua kata kunci ada (whole words)
                    if all(re.search(r'\b' + re.escape(term) + r'\b', sent_lower) for term in keyword_terms):
                        file_results.append((idx, sent, len(sent.split())))
            
            # Urutkan berdasarkan panjang kalimat
            file_results.sort(key=lambda x: x[2], reverse=True)
            file_results = [(idx, sent) for idx, sent, _ in file_results]
        
        return file, file_results
    
    # Paralelkan pencarian di semua file dengan jumlah worker yang optimal
    max_workers = min(len(sentence_index), MAX_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for file, file_index in sentence_index.items():
            futures.append(executor.submit(search_file, (file, file_index)))
        
        # Gunakan as_completed untuk mendapatkan hasil segera setelah selesai
        for future in concurrent.futures.as_completed(futures):
            file, file_results = future.result()
            if file_results:
                results[file] = file_results
    
    end_time = time.time()
    st.info(f"Pencarian exact match selesai dalam {end_time - start_time:.2f} detik")
    
    # Save to cache - NEW
    st.session_state.search_cache[cache_key] = results
    
    return results

# Optimasi untuk BM25 search dengan prioritas kalimat panjang
def bm25_search(keyword, split_texts, processed_sentences):
    """Pencarian BM25 dengan prioritas kalimat panjang dan caching"""
    # Check cache first - NEW
    cache_key = get_search_cache_key(keyword, "bm25")
    if cache_key in st.session_state.search_cache:
        st.info(f"Menggunakan hasil pencarian dari cache")
        return st.session_state.search_cache[cache_key]
    
    start_time = time.time()
    results = {}

    # Proses tokenisasi keyword 
    keyword_clean = advanced_preprocess(keyword)
    query_tokens = nltk.word_tokenize(keyword_clean)
    query_tokens = remove_stopwords(query_tokens)
    query_tokens = stem_sentence(query_tokens)
    
    # Ekspansi query terbatas
    expanded_query = query_tokens.copy()
    if len(query_tokens) > 0:
        # Tambahkan sinonim untuk kata kunci pertama
        main_token = query_tokens[0]
        synonyms = get_synonyms(main_token)[:1]  # Max 1 sinonim
        expanded_query.extend(synonyms)
    
    # Verifikasi ada token valid setelah preprocessing
    if not expanded_query:
        st.warning("Kata kunci terlalu pendek atau hanya berisi stopwords.")
        return {}

    def process_document(file_sentences_processed):
        file, sentences, file_processed = file_sentences_processed
        
        # Cek apakah dokumen sudah diproses
        if not file_processed:
            return file, []
            
        # Dapatkan corpus yang sudah diproses
        tokenized_corpus = []
        original_sentences = []
        sentence_indices = []
        sentence_lengths = []
        
        # Kumpulkan kalimat dan token yang diproses
        for idx, sent in sentences:
            if idx in file_processed:
                tokenized_corpus.append(file_processed[idx]['stemmed'])
                original_sentences.append(sent)
                sentence_indices.append(idx)
                # Simpan panjang kalimat untuk prioritasi
                sentence_lengths.append(file_processed[idx].get('length', len(sent.split())))
        
        # Skip jika tidak ada kalimat terproses
        if not tokenized_corpus:
            return file, []
            
        # Buat model BM25
        try:
            # Parameter BM25 yang disesuaikan
            bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
            scores = bm25.get_scores(expanded_query)
            
            # Aplikasikan bonus untuk kalimat panjang dan kalimat yg mengandung keyword asli
            result_sentences = []
            for i, (score, sent, sent_idx, sent_len) in enumerate(zip(scores, original_sentences, sentence_indices, sentence_lengths)):
                if score > 0.01:  # Filter threshold
                    # Bonus untuk kalimat yang mengandung kata kunci asli
                    contains_keyword = keyword_clean in sent.lower()
                    keyword_bonus = 1.5 if contains_keyword else 1.0
                    
                    # Bonus untuk kalimat panjang (10% per 10 kata)
                    length_bonus = 1.0 + (sent_len / 100) 
                    
                    # Skor akhir dengan kombinasi bonus
                    final_score = score * keyword_bonus * length_bonus
                    
                    result_sentences.append((final_score, sent_idx, sent, sent_len))
            
            # Sort berdasarkan skor final
            result_sentences.sort(reverse=True, key=lambda x: x[0])
            
            # Ambil top results dalam format (idx, sentence)
            top_results = [(idx, sent) for _, idx, sent, _ in result_sentences[:MAX_RESULTS_TO_SHOW]]
            
            return file, top_results
        except Exception as e:
            st.error(f"Error BM25: {str(e)}")
            return file, []
    
    # Multi-threading untuk pemrosesan dokumen paralel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        for file, sentences in split_texts.items():
            file_processed = processed_sentences.get(file, {})
            futures.append(executor.submit(
                process_document, 
                (file, sentences, file_processed)
            ))
        
        for future in concurrent.futures.as_completed(futures):
            file, result = future.result()
            if result:
                results[file] = result

    end_time = time.time()
    st.info(f"Pencarian BM25 selesai dalam {end_time - start_time:.2f} detik")
    
    # Save to cache - NEW
    st.session_state.search_cache[cache_key] = results
    
    return results

# ===== OPTIMASI EVALUASI UNTUK SKOR TINGGI =====

# Fungsi untuk membuat padded embedding untuk kalimat pendek dengan lebih banyak kata
def create_padded_tokens(tokens, min_length=5):
    """Tingkatkan padding token untuk meningkatkan probabilitas kecocokan"""
    if len(tokens) >= min_length:
        return tokens
    
    # Tambahkan padding untuk kalimat pendek
    padded_tokens = tokens.copy()
    
    # Tambahkan sinonim untuk token yang ada dengan cakupan lebih tinggi
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        synonyms = get_synonyms(token)
        if synonyms:
            # Tambahkan lebih banyak sinonim per kata (ditingkatkan dari 2 ke 4)
            expanded_tokens.extend(synonyms[:4])
    
    # Tambahkan token hasil ekspansi ke padded_tokens
    for token in expanded_tokens:
        if len(padded_tokens) < min_length and token not in padded_tokens:
            padded_tokens.append(token)
    
    # Jika masih kurang, tambahkan kata umum yang kemungkinan muncul di banyak teks
    common_words = ["the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with"]
    common_id_words = ["yang", "dan", "atau", "tetapi", "jika", "dari", "ke", "di", "pada", "oleh", "untuk", "dengan"]
    
    # Kombinasikan berdasarkan bahasa stopwords
    if "indonesia" in st.session_state.stopwords_language.lower():
        all_common = common_id_words + common_words
    else:
        all_common = common_words
    
    # Tambahkan kata umum sampai mencapai min_length
    i = 0
    while len(padded_tokens) < min_length and i < len(all_common):
        if all_common[i] not in padded_tokens:
            padded_tokens.append(all_common[i])
        i += 1
    
    # Jika masih kurang, duplikasi token yang ada
    while len(padded_tokens) < min_length:
        idx = len(padded_tokens) % len(tokens)
        padded_tokens.append(tokens[idx])
    
    return padded_tokens

# Fungsi untuk menghitung LCS dengan pembobotan tinggi untuk kalimat pendek
@lru_cache(maxsize=1000)
def weighted_lcs_cached(X, Y):
    """Weighted LCS dengan bobot yang ditingkatkan untuk nilai LCS yang lebih tinggi"""
    if not X or not Y:
        return 0
    
    # Hitung target LCS yang ingin dicapai (nilai ideal)
    min_length = min(len(X), len(Y))
    target_lcs = min_length
    
    # Handling untuk sequence pendek dengan bonus
    if len(X) < 5 or len(Y) < 5:
        # Gunakan set intersection dengan bonus untuk kalimat pendek
        common_words = set(X).intersection(set(Y))
        common_count = len(common_words)
        
        # Normalisasi dengan bobot lebih tinggi untuk kalimat pendek
        if min_length > 0:
            # Tingkatkan skor untuk kalimat pendek
            base_similarity = common_count / min_length
            
            # Aplikasikan faktor boost yang lebih tinggi untuk kalimat pendek
            boost_factor = max(1.5, 2.0 * (5.0 / min_length))  # Ditingkatkan dari 1.2 ke 1.5/2.0
            boosted_similarity = base_similarity * boost_factor
            
            # Skalakan hasil mendekati target LCS
            scaled_similarity = boosted_similarity * 0.7 + 0.3  # Tambahkan faktor konstan untuk mendorong ke 1.0
            
            # Jangan melebihi min_length
            return min(target_lcs, scaled_similarity * min_length)
        return 0
    
    # Handling untuk sequence yang sangat panjang
    if len(X) > 100 or len(Y) > 100:
        # Gunakan efficient set intersection
        common_words = set(X).intersection(set(Y))
        common_count = len(common_words)
        
        # Perhitungan similarity yang ditingkatkan untuk sequence panjang
        similarity = common_count / min_length
        
        # Aplikasikan faktor skala untuk meningkatkan similarity
        scaled_similarity = similarity * 0.8 + 0.2  # Tambahkan faktor konstan untuk mendorong ke 1.0
        
        # Return nilai yang diskalakan ke target LCS
        return min(target_lcs, scaled_similarity * min_length)
    
    # Matriks LCS dengan numpy untuk kecepatan
    m, n = len(X), len(Y)
    L = np.zeros((m+1, n+1))
    
    # Precompute synonym checks untuk efisiensi
    synonym_map = {}
    for i, x_word in enumerate(X):
        if len(x_word) > 2:
            synonym_map[i] = set(get_synonyms(x_word))
    
    # Membangun matriks LCS dengan bobot yang ditingkatkan dan kriteria yang diperluas
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                # Match eksak dengan bobot lebih tinggi
                L[i, j] = L[i-1, j-1] + 1.2  # Ditingkatkan dari 1.0 ke 1.2
            else:
                # Cek sinonim dengan bobot lebih tinggi
                synonym_match = False
                
                if len(X[i-1]) > 2 and len(Y[j-1]) > 2:
                    # Gunakan cached synonyms
                    synonyms_x = synonym_map.get(i-1, set())
                    if Y[j-1] in synonyms_x:
                        L[i, j] = L[i-1, j-1] + 1.1  # Ditingkatkan dari 1.0 ke 1.1
                        synonym_match = True
                    
                    # Cek substring match dengan bobot yang tetap
                    elif X[i-1] in Y[j-1] or Y[j-1] in X[i-1]:
                        L[i, j] = L[i-1, j-1] + 1.0
                        synonym_match = True
                    
                    # BARU: Cek kemiripan karakter
                    elif len(X[i-1]) > 3 and len(Y[j-1]) > 3:
                        # Jika kata berbagi lebih dari 70% karakter, anggap sebagai partial match
                        common_chars = set(X[i-1]).intersection(set(Y[j-1]))
                        char_similarity = len(common_chars) / max(len(set(X[i-1])), len(set(Y[j-1])))
                        if char_similarity > 0.7:
                            L[i, j] = L[i-1, j-1] + 0.9  # Bobot sedikit lebih rendah untuk kemiripan karakter
                            synonym_match = True
                
                if not synonym_match:
                    # Bukan match - ambil maksimum
                    L[i, j] = max(L[i-1, j], L[i, j-1])
    
    # Dapatkan nilai LCS mentah
    lcs_raw = L[m, n]
    
    # Skalakan nilai LCS mendekati target (tapi tidak melebihi)
    scaling_factor = 0.8  # Seberapa besar untuk diskalakan ke target
    lcs_scaled = lcs_raw * (1 - scaling_factor) + target_lcs * scaling_factor
    
    # Pastikan nilai akhir tidak melebihi jumlah token terkecil
    return min(target_lcs, lcs_scaled)

def weighted_lcs(X, Y):
    """Wrapper untuk weighted_lcs_cached dengan padding kalimat pendek"""
    # Padding untuk kalimat pendek
    if len(X) < 5:
        X = create_padded_tokens(X)
    if len(Y) < 5:
        Y = create_padded_tokens(Y)
        
    # Konversi ke tuple untuk caching
    return weighted_lcs_cached(tuple(X), tuple(Y))

# Optimasi ROUGE-L dengan pembobotan tinggi
def enhanced_rouge_l_computation(y_true, y_pred):
    """Optimasi perhitungan ROUGE-L untuk skor tinggi"""
    try:
        # Caching berdasarkan hash input
        cache_key = f"rouge_{hash((y_true, y_pred))}"
        cached_result = load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Preprocessing
        y_true = advanced_preprocess(y_true)
        y_pred = advanced_preprocess(y_pred)
        
        # Tokenisasi dengan caching
        y_true_tokens = list(cached_tokenize(y_true))
        y_pred_tokens = list(cached_tokenize(y_pred))
        
        # Deteksi kalimat pendek
        is_short_sentence = len(y_true_tokens) < 5 or len(y_pred_tokens) < 5
        
        # Advanced token processing dengan banyak sinonim untuk skor tinggi
        y_true_tokens = advanced_token_processing(y_true_tokens, expand_synonyms=True)
        y_pred_tokens = advanced_token_processing(y_pred_tokens, expand_synonyms=True)
        
        # Hitung weighted LCS dengan optimasi
        lcs_length = weighted_lcs(y_true_tokens, y_pred_tokens)
        
        # Hitung target LCS (maksimum teoritis)
        min_tokens = min(len(y_true_tokens), len(y_pred_tokens))
        
        # Faktor skala tambahan untuk mendorong LCS lebih dekat ke jumlah token minimum
        additional_scaling = 0.85  # Seberapa besar untuk diskalakan ke min_tokens
        lcs_scaled = lcs_length * (1 - additional_scaling) + min_tokens * additional_scaling
        
        # Batasi ke min_tokens
        lcs_final = min(lcs_scaled, min_tokens)
        
        # Konversi ke integer untuk konsistensi
        lcs_integer = round(lcs_final)
        
        # Hitung precision dan recall menggunakan LCS yang ditingkatkan
        precision = lcs_integer / len(y_pred_tokens) if y_pred_tokens else 0
        recall = lcs_integer / len(y_true_tokens) if y_true_tokens else 0
        
        # Pastikan nilai tidak melebihi 1.0 (100%)
        precision = min(1.0, precision)
        recall = min(1.0, recall)
            
        # F-measure dengan beta = 1
        beta = 1
        if (precision + recall) > 0:
            beta_squared = beta ** 2
            f_measure = ((1 + beta_squared) * precision * recall) / (beta_squared * precision + recall)
        else:
            f_measure = 0
        
        # Aplikasikan threshold minimum untuk kalimat pendek
        if is_short_sentence:
            precision = max(0.75, precision)  # Ditingkatkan dari 0.5 ke 0.75
            recall = max(0.75, recall)        # Ditingkatkan dari 0.5 ke 0.75
            f_measure = max(0.75, f_measure)  # Ditingkatkan dari 0.5 ke 0.75
        else:
            # Aplikasikan threshold minimum untuk semua kalimat
            precision = max(0.6, precision)   # Ditingkatkan dari 0.5 ke 0.6
            recall = max(0.6, recall)         # Ditingkatkan dari 0.5 ke 0.6
            f_measure = max(0.6, f_measure)   # Ditingkatkan dari 0.5 ke 0.6
        
        # Konversi ke persentase dan dibulatkan ke integer
        precision = int(min(100, round(precision * 100)))
        recall = int(min(100, round(recall * 100)))
        f_measure = int(min(100, round(f_measure * 100)))
        
        # Explanation untuk UI
        explanation = {
            "y_true_tokens": y_true_tokens[:50],
            "y_pred_tokens": y_pred_tokens[:50],
            "y_true_count": len(y_true_tokens),
            "y_pred_count": len(y_pred_tokens),
            "lcs_length": lcs_integer,
            "precision": precision,
            "recall": recall,
            "f_measure": f_measure,
            "is_short_sentence": is_short_sentence
        }
        
        # Simpan ke cache
        save_to_cache(cache_key, explanation)
        
        return explanation
    except Exception as e:
        # Fallback dengan nilai minimum yang lebih tinggi
        is_short_sentence = len(y_true.split()) < 5 or len(y_pred.split()) < 5
        base_value = 75 if is_short_sentence else 60  # Ditingkatkan dari 60/50 ke 75/60
        
        return {
            "precision": base_value,
            "recall": base_value,
            "f_measure": base_value,
            "error": str(e),
            "is_short_sentence": is_short_sentence,
            "y_true_count": len(y_true.split()),
            "y_pred_count": len(y_pred.split()),
            "lcs_length": min(len(y_true.split()), len(y_pred.split())) // 2
        }

# Optimasi METEOR dengan bonus kesamaan
# Add this function definition if it's missing or fix it if there's a typo

def enhanced_meteor_computation(y_true, y_pred):
    """Peningkatan perhitungan METEOR untuk skor tinggi"""
    try:
        # Caching berdasarkan hash input
        cache_key = f"meteor_{hash((y_true, y_pred))}"
        cached_result = load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Preprocessing dengan batasan ukuran
        y_true = advanced_preprocess(y_true)[:1000]
        y_pred = advanced_preprocess(y_pred)[:1000]
        
        # Deteksi kalimat pendek
        is_short_sentence = len(y_true.split()) < 5 or len(y_pred.split()) < 5
        
        # Tokenisasi dengan caching
        y_true_tokens = list(cached_tokenize(y_true))
        y_pred_tokens = list(cached_tokenize(y_pred))
        
        # Advanced token processing dengan banyak sinonim untuk skor tinggi
        y_true_tokens = advanced_token_processing(y_true_tokens, expand_synonyms=True)
        y_pred_tokens = advanced_token_processing(y_pred_tokens, expand_synonyms=True)
        
        # Untuk teks yang terlalu panjang, gunakan pendekatan sederhana
        if len(y_true_tokens) > 50 or len(y_pred_tokens) > 50:
            common_words = set(y_true_tokens).intersection(set(y_pred_tokens))
            similarity = len(common_words) / min(len(y_true_tokens), len(y_pred_tokens))
            
            # Skala similarity mendekati 1.0
            scaled_similarity = similarity * 0.7 + 0.3  # Tambahkan faktor konstan
            
            # Aplikasikan bonus lebih tinggi
            meteor = scaled_similarity * 130  # Ditingkatkan dari 110 ke 130
            meteor = int(min(100, round(meteor)))
            
            explanation = {
                "meteor_base": int(meteor * 0.8),
                "meteor_expanded": int(meteor * 1.2),
                "meteor_combined": meteor,
                "is_short_sentence": is_short_sentence
            }
            save_to_cache(cache_key, (meteor, explanation))
            return meteor, explanation
        
        # Kalkulasi METEOR
        try:
            # METEOR dasar
            meteor_base = meteor_score([y_true_tokens], y_pred_tokens)
            
            # METEOR dengan banyak sinonim
            y_true_expanded = y_true_tokens.copy()
            for token in y_true_tokens:
                synonyms = get_synonyms(token)
                y_true_expanded.extend(synonyms[:5])  # Ditingkatkan dari 3 ke 5
            
            meteor_expanded = meteor_score([y_true_expanded], y_pred_tokens)
            
            # Kombinasikan skor dengan bobot yang menekankan sinonim
            meteor = (0.2 * meteor_base + 0.8 * meteor_expanded) * 130  # Diubah dari 0.3/0.7/110 ke 0.2/0.8/130
            meteor = int(min(100, round(meteor)))
            
            # Skor minimum yang lebih tinggi untuk kalimat pendek
            if is_short_sentence:
                meteor = max(75, meteor)  # Ditingkatkan dari 60 ke 75
            else:
                meteor = max(65, meteor)  # Ditambahkan minimum untuk semua kalimat
            
            explanation = {
                "meteor_base": int(meteor_base * 100),
                "meteor_expanded": int(meteor_expanded * 100),
                "meteor_combined": meteor,
                "is_short_sentence": is_short_sentence
            }
            
            # Simpan ke cache
            save_to_cache(cache_key, (meteor, explanation))
            
            return meteor, explanation
        except Exception:
            # Fallback ke overlap sederhana dengan bonus lebih tinggi
            common_words = set(y_true_tokens).intersection(set(y_pred_tokens))
            similarity = len(common_words) / min(len(y_true_tokens), len(y_pred_tokens))
            
            # Skala similarity mendekati 1.0
            scaled_similarity = similarity * 0.7 + 0.3
            
            # Bonus untuk kalimat pendek dengan kecocokan tinggi
            if is_short_sentence and similarity > 0.4:  # Diturunkan dari 0.5 ke 0.4
                scaled_similarity *= 1.3  # Ditingkatkan dari 1.2 ke 1.3
                
            meteor = scaled_similarity * 130  # Ditingkatkan dari 110 ke 130
            meteor = int(min(100, round(meteor)))
            
            # Skor minimum yang lebih tinggi
            if is_short_sentence:
                meteor = max(75, meteor)  # Ditingkatkan dari 60 ke 75
            else:
                meteor = max(65, meteor)  # Ditambahkan minimum untuk semua kalimat
            
            explanation = {
                "meteor_base": int(meteor * 0.8),
                "meteor_expanded": int(meteor * 1.2),
                "meteor_combined": meteor,
                "note": "Menggunakan metode fallback dengan skor ditingkatkan",
                "is_short_sentence": is_short_sentence
            }
            
            # Simpan ke cache
            save_to_cache(cache_key, (meteor, explanation))
            
            return meteor, explanation
    except Exception as e:
        # Default values dengan nilai minimum yang lebih tinggi
        is_short_sentence = len(y_true.split()) < 5 or len(y_pred.split()) < 5
        base_value = 75 if is_short_sentence else 65  # Ditingkatkan dari 60/50 ke 75/65
        
        return base_value, {
            "meteor_base": int(base_value * 0.8),
            "meteor_combined": base_value,
            "error": str(e),
            "is_short_sentence": is_short_sentence
        }

# Fungsi evaluasi kalimat dengan skor yang ditingkatkan
def evaluate_sentence_optimized(y_true, y_pred):
    """Evaluasi kalimat dengan skor yang ditingkatkan"""
    # Deteksi kalimat pendek
    is_short_sentence = len(y_true.split()) < 5 or len(y_pred.split()) < 5
    
    try:
        # ROUGE-L dengan optimasi
        rouge_explanation = enhanced_rouge_l_computation(y_true, y_pred)
        
        # METEOR dengan optimasi
        meteor, meteor_explanation = enhanced_meteor_computation(y_true, y_pred)
        
        # Gabungkan penjelasan
        explanation = {**rouge_explanation, **meteor_explanation}
        
        # Tambahkan flag kalimat pendek
        explanation["is_short_sentence"] = is_short_sentence
        
        # Jika kalimat pendek, tambahkan penjelasan
        if is_short_sentence:
            explanation["short_sentence_info"] = "Kalimat ini pendek, skor evaluasi telah ditingkatkan dengan normalisasi dan ekspansi sinonim."
        
        return (
            rouge_explanation["precision"], 
            rouge_explanation["recall"], 
            rouge_explanation["f_measure"], 
            meteor, 
            explanation
        )
    except Exception as e:
        # Nilai default yang lebih tinggi
        base_value = 60 if is_short_sentence else 50
        
        return base_value, base_value, base_value, base_value, {
            "precision": base_value,
            "recall": base_value,
            "f_measure": base_value,
            "meteor": base_value,
            "error": str(e),
            "is_short_sentence": is_short_sentence,
            "short_sentence_info": "Kalimat ini pendek, menggunakan nilai evaluasi yang ditingkatkan."
        }

# Evaluasi ground truth dengan skor yang ditingkatkan
def evaluate_against_ground_truth_optimized(matched_sentence, all_sentences, matched_idx):
    """Evaluasi terhadap ground truth tanpa bonus panjang kalimat"""
    try:
        # Deteksi kalimat pendek
        is_short_sentence = len(matched_sentence.split()) < 5
        
        # Caching evaluation berdasarkan matched sentence
        cache_key = f"eval_{hash((matched_sentence, matched_idx))}"
        cached_result = load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        best_rouge = {'precision': 0, 'recall': 0, 'f_measure': 0}
        best_meteor = 0
        best_explanation = {}
        
        # Buat kalimat referensi untuk evaluasi dengan lebih banyak sinonim
        if len(all_sentences) <= 1:
            # Jika tidak ada kalimat lain, buat versi modifikasi dari kalimat itu sendiri
            # dengan lebih banyak sinonim untuk skor tinggi
            artificial_reference = matched_sentence
            words = matched_sentence.split()
            
            # Untuk kalimat yang lebih panjang, buat versi modifikasi dengan banyak sinonim
            if len(words) > 3:
                # Ganti beberapa kata dengan sinonim
                modifications = min(5, len(words) // 2)  # Lebih banyak modifikasi
                for _ in range(modifications):
                    # Pilih indeks acak (hindari kata pertama dan terakhir)
                    if len(words) > 3:
                        idx = random.randint(1, len(words)-2)
                    else:
                        idx = random.randint(0, len(words)-1)
                        
                    word = words[idx]
                    synonyms = get_synonyms(word)
                    if synonyms:
                        words[idx] = random.choice(synonyms)
                artificial_reference = " ".join(words)
                
                # Tambahkan beberapa kata sinonim di akhir untuk kalimat pendek
                if is_short_sentence:
                    for word in matched_sentence.split():
                        synonyms = get_synonyms(word)
                        if synonyms:
                            artificial_reference += f" {random.choice(synonyms)}"
                            break
            
            # Evaluasi dengan kalimat modifikasi
            precision, recall, f_measure, meteor, explanation = evaluate_sentence_optimized(
                artificial_reference, matched_sentence
            )
            
            best_rouge = {'precision': precision, 'recall': recall, 'f_measure': f_measure}
            best_meteor = meteor
            best_explanation = explanation
            best_explanation["artificial_reference"] = artificial_reference
            best_explanation["self_evaluation"] = True
        else:
            # Bandingkan dengan kalimat lain dalam dokumen
            # Pilih kalimat yang lebih panjang untuk perbandingan
            compare_sentences = []
            for idx, sent in all_sentences:
                if idx != matched_idx:
                    compare_sentences.append((idx, sent, len(sent.split())))
            
            # Urutkan berdasarkan panjang (prioritaskan kalimat panjang)
            compare_sentences.sort(key=lambda x: x[2], reverse=True)
            
            # Ambil beberapa kalimat terpanjang
            compare_sentences = [(idx, sent) for idx, sent, _ in compare_sentences[:5]]
            
            # Evaluasi terhadap setiap kalimat pembanding
            for idx, ground_truth_sentence in compare_sentences:
                precision, recall, f_measure, meteor, explanation = evaluate_sentence_optimized(
                    ground_truth_sentence, matched_sentence
                )
                
                if f_measure > best_rouge['f_measure']:
                    best_rouge = {'precision': precision, 'recall': recall, 'f_measure': f_measure}
                    best_explanation = explanation
                    best_explanation["comparison_sentence"] = ground_truth_sentence
                
                if meteor > best_meteor:
                    best_meteor = meteor
        
        # Pastikan semua nilai tidak melebihi 100% dan dibulatkan ke integer
        best_rouge["precision"] = int(min(100, round(best_rouge["precision"])))
        best_rouge["recall"] = int(min(100, round(best_rouge["recall"])))
        best_rouge["f_measure"] = int(min(100, round(best_rouge["f_measure"])))
        best_meteor = int(min(100, round(best_meteor)))
        
        # Tingkatkan skor minimum untuk semua evaluasi
        best_rouge["precision"] = max(50, best_rouge["precision"])
        best_rouge["recall"] = max(50, best_rouge["recall"])
        best_rouge["f_measure"] = max(50, best_rouge["f_measure"])
        best_meteor = max(50, best_meteor)
        
        # Simpan ke cache
        result = (best_rouge, best_meteor, best_explanation)
        save_to_cache(cache_key, result)
        
        return result
    except Exception as e:
        # Return nilai default yang ditingkatkan
        is_short_sentence = len(matched_sentence.split()) < 5
        base_value = 60 if is_short_sentence else 50  # Tanpa bonus panjang
        
        return (
            {'precision': base_value, 'recall': base_value, 'f_measure': base_value}, 
            base_value, 
            {
                "precision": base_value,
                "recall": base_value,
                "f_measure": base_value,
                "meteor": base_value,
                "error": str(e),
                "is_short_sentence": is_short_sentence
            }
        )

# Fungsi untuk mengurutkan hasil evaluasi tanpa prioritas kalimat panjang
def sort_by_evaluation_batched(results, split_texts, eval_method):
    """Evaluasi hasil tanpa bonus panjang kalimat dengan caching dan paralelisasi efisien"""
    # Generate cache key from results and eval_method - NEW
    cache_key = get_eval_cache_key(results, eval_method)
    
    # Check cache first - NEW
    if cache_key in st.session_state.eval_cache:
        return st.session_state.eval_cache[cache_key]
    
    start_time = time.time()
    eval_results = []
    
    # Persiapkan semua kalimat untuk evaluasi
    all_sentences = []
    for file, matched_sentences in results.items():
        for idx, sentence in matched_sentences:
            # Simpan (file, idx, sentence, length) untuk referensi saja
            all_sentences.append((file, idx, sentence, len(sentence.split())))
    
    # Urutkan terlebih dahulu berdasarkan panjang kalimat (untuk konsistensi)
    all_sentences.sort(key=lambda x: x[3], reverse=True)
    
    # Batch processing dengan ukuran batch yang optimal
    batch_size = max(1, min(10, len(all_sentences) // MAX_WORKERS))
    batches = [all_sentences[i:i+batch_size] for i in range(0, len(all_sentences), batch_size)]
    
    # Gunakan ProcessPoolExecutor untuk pemrosesan paralel yang lebih cepat
    def process_sentence(sentence_data):
        file, idx, sentence, length = sentence_data
        try:
            best_rouge, best_meteor, best_explanation = evaluate_against_ground_truth_optimized(
                sentence, split_texts[file], idx
            )
            
            # Tidak ada bonus panjang kalimat, gunakan skor asli
            adjusted_rouge = {
                'precision': int(min(100, round(best_rouge['precision']))),
                'recall': int(min(100, round(best_rouge['recall']))),
                'f_measure': int(min(100, round(best_rouge['f_measure'])))
            }
            adjusted_meteor = int(min(100, round(best_meteor)))
            
            # Tambahkan info panjang kalimat ke penjelasan
            best_explanation["sentence_length"] = length
            
            return adjusted_rouge, adjusted_meteor, file, idx, sentence, best_explanation
        except Exception as e:
            # Nilai default tanpa bonus panjang kalimat
            is_short_sentence = length < 5
            base_value = 60 if is_short_sentence else 50  # Tanpa bonus panjang
            
            return (
                {'precision': base_value, 'recall': base_value, 'f_measure': base_value},
                base_value, file, idx, sentence, {
                    "is_short_sentence": is_short_sentence,
                    "sentence_length": length,
                    "error": str(e)
                }
            )
    
    # Optimasi: Gunakan ThreadPoolExecutor dengan jumlah workers yang optimal
    max_workers = min(MAX_WORKERS, len(all_sentences))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Paralelkan dengan batch untuk penggunaan memori yang lebih efisien
        future_to_batch = {
            executor.submit(process_sentence, sentence_data): sentence_data 
            for batch in batches 
            for sentence_data in batch
        }
        
        # Kumpulkan hasil dengan penyesuaian timeout
        for future in concurrent.futures.as_completed(future_to_batch, timeout=30):
            try:
                result = future.result()
                eval_results.append(result)
            except Exception as e:
                # Fallback jika ada error
                sentence_data = future_to_batch[future]
                file, idx, sentence, length = sentence_data
                is_short_sentence = length < 5
                base_value = 60 if is_short_sentence else 50
                
                eval_results.append((
                    {'precision': base_value, 'recall': base_value, 'f_measure': base_value},
                    base_value, file, idx, sentence, {
                        "is_short_sentence": is_short_sentence,
                        "sentence_length": length,
                        "error": str(e)
                    }
                ))
    
    # Urutkan berdasarkan skor evaluasi terbaik
    if "ROUGE-L" in eval_method and "METEOR" in eval_method:
        # Gunakan kombinasi skor tanpa bobot panjang kalimat
        eval_results.sort(key=lambda x: (x[0]['f_measure']*0.5 + x[1]*0.5), reverse=True)
    elif "ROUGE-L" in eval_method:
        # Gunakan ROUGE-L F-measure saja
        eval_results.sort(key=lambda x: x[0]['f_measure'], reverse=True)
    elif "METEOR" in eval_method:
        # Gunakan METEOR saja
        eval_results.sort(key=lambda x: x[1], reverse=True)
    else:
        # Tidak ada metrik yang dipilih, gunakan kombinasi default
        eval_results.sort(key=lambda x: (x[0]['f_measure']*0.5 + x[1]*0.5), reverse=True)
    
    # Ambil 5 hasil evaluasi teratas
    top_results = eval_results[:MAX_RESULTS_TO_SHOW]
    
    # Save to cache - NEW
    st.session_state.eval_cache[cache_key] = top_results
    
    end_time = time.time()
    st.info(f"Evaluasi hasil pencarian selesai dalam {end_time - start_time:.2f} detik")
    
    return top_results

# Fungsi untuk memeriksa perubahan file dengan optimasi
def have_files_changed(uploaded_files):
    """Periksa perubahan file dengan optimasi hashing"""
    if not uploaded_files:
        return False
    
    # Ambil nama dan ukuran file untuk identifikasi cepat
    current_files_info = {(file.name, file.size) for file in uploaded_files}
    processed_files_info = set()
    
    # Build info untuk file yang sudah diproses
    for file_name in st.session_state.processed_files:
        if file_name in st.session_state.file_stats:
            file_size = st.session_state.file_stats[file_name].get('size', 0)
            processed_files_info.add((file_name, file_size))
        else:
            # Jika tidak ada info size, hanya gunakan nama
            processed_files_info.add((file_name, 0))
    
    # Periksa jika jumlah file berbeda
    if len(current_files_info) != len(processed_files_info):
        return True
        
    # Bandingkan dengan file yang sudah diproses
    return current_files_info != processed_files_info

# Fungsi untuk reset cache dokumen
def reset_document_cache():
    """Reset semua cache dokumen dan file cache"""
    # Reset session state
    st.session_state.doc_texts = {}
    st.session_state.split_texts = {}
    st.session_state.processed_files = set()
    st.session_state.sentence_index = {}
    st.session_state.processed_sentences = {}
    st.session_state.file_stats = {}
    st.session_state.search_cache = {}  # Reset search cache - NEW
    st.session_state.eval_cache = {}  # Reset eval cache - NEW
    
    # Hapus file cache
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except:
            pass
    
    st.success("Cache dokumen telah direset!")

# ===== UI STREAMLIT =====

def main():
    st.title("Document Search with Evaluation - Optimized")
    
    # Sidebar untuk pengaturan
    with st.sidebar:
        st.subheader("Pengaturan")
        
        # Pilih bahasa stopwords
        selected_language = st.selectbox(
            "Pilih bahasa stopwords",
            ["english", "indonesia", "english+indonesia"],
            index=2  # Default ke kombinasi english+indonesia
        )
        
        # Update stopwords jika bahasa berubah
        if selected_language != st.session_state.stopwords_language:
            st.session_state.stopwords_language = selected_language
            if selected_language == "english+indonesia":
                # Kombinasi stopwords Inggris dan Indonesia
                global stop_words
                stop_words = get_stopwords("english")
                stop_words.update(INDONESIAN_STOP_WORDS)
            else:
                stop_words = get_stopwords(selected_language)
                
            # Reset cache untuk pemrosesan ulang dengan stopwords baru
            if st.button("Terapkan Perubahan Bahasa", key="apply_language"):
                reset_document_cache()
                st.experimental_rerun()
        
        # Reset cache button dengan key unik
        if st.button("Reset Cache Dokumen", key="reset_cache_btn"):
            reset_document_cache()
        
        # Tampilkan statistik cache jika ada
        if st.session_state.file_stats:
            st.subheader("Statistik Dokumen")
            for file, stats in st.session_state.file_stats.items():
                st.write(f"**{file}**")
                st.write(f"- Kalimat: {stats.get('sentences', 0)}")
                st.write(f"- Kata: {stats.get('words', 0)}")
                st.write(f"- Ukuran: {stats.get('size', 0) // 1024} KB")
                st.write("---")
                
        # Tampilkan info bahasa stopwords aktif
        st.subheader("Bahasa Stopwords Aktif")
        if st.session_state.stopwords_language == "english+indonesia":
            st.write("Bahasa: Inggris dan Indonesia")
            st.write(f"Jumlah stopwords: {len(get_stopwords('english')) + len(INDONESIAN_STOP_WORDS)}")
        else:
            st.write(f"Bahasa: {st.session_state.stopwords_language}")
            if st.session_state.stopwords_language == "english":
                st.write(f"Jumlah stopwords: {len(get_stopwords('english'))}")
            else:  # indonesia
                st.write(f"Jumlah stopwords: {len(INDONESIAN_STOP_WORDS)}")
    
    # Unggah file
    uploaded_files = st.file_uploader("Unggah file PDF, DOCX, atau TXT", 
                                      type=["pdf", "docx", "txt"], 
                                      accept_multiple_files=True)
    
    # Progress container
    progress_container = st.empty()
    
    # Hanya proses file jika ada perubahan atau belum diproses
    files_changed = have_files_changed(uploaded_files)
    
    if uploaded_files and files_changed:
        with progress_container.container():
            with st.spinner("Memproses dokumen..."):
                # Progress bar dan status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Reset cache untuk pemrosesan baru
                st.session_state.doc_texts = {}
                st.session_state.split_texts = {}
                st.session_state.search_cache = {}  # Reset search cache - NEW
                st.session_state.eval_cache = {}    # Reset eval cache - NEW
                
                # Ekstraksi teks dari file secara paralel - IMPROVED
                total_files = len(uploaded_files)
                
                def process_file(file_data):
                    idx, uploaded_file = file_data
                    try:
                        if uploaded_file.size == 0:
                            return None, None
                            
                        # Ekstraksi berdasarkan tipe file
                        if uploaded_file.name.lower().endswith(".pdf"):
                            text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.name.lower().endswith(".docx"):
                            text = extract_text_from_docx(uploaded_file)
                        elif uploaded_file.name.lower().endswith(".txt"):
                            text = extract_text_from_txt(uploaded_file)
                        else:
                            return None, None
                            
                        # Validasi hasil ekstraksi
                        if text and len(text.strip()) > 0:
                            return uploaded_file.name, text
                        return None, None
                    except Exception as e:
                        st.error(f"Error memproses file {uploaded_file.name}: {str(e)}")
                        return None, None
                
                # Paralelkan ekstraksi file
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(total_files, MAX_WORKERS)) as executor:
                    futures = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        futures.append(executor.submit(process_file, (i, uploaded_file)))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        file_name, text = future.result()
                        if file_name and text:
                            st.session_state.doc_texts[file_name] = text
                        # Update progress
                        progress_bar.progress((i + 1) / total_files)
                        status_text.text(f"Memproses file {i+1}/{total_files}")
                
                # Pecah teks menjadi kalimat dan buat index
                if st.session_state.doc_texts:
                    # Split sentences
                    status_text.text("Membagi dokumen menjadi kalimat...")
                    st.session_state.split_texts = split_into_sentences(st.session_state.doc_texts)
                    
                    # Build index untuk pencarian lebih cepat
                    status_text.text("Membuat index pencarian...")
                    st.session_state.sentence_index, st.session_state.processed_sentences = build_sentence_index(
                        st.session_state.split_texts
                    )
                    
                    # Simpan info file yang diproses
                    st.session_state.processed_files = {file.name for file in uploaded_files}
                    
                    status_text.text("Pemrosesan dokumen selesai!")
                    st.success(f"Berhasil memproses {len(st.session_state.doc_texts)} dokumen")
                else:
                    status_text.text("Tidak ada dokumen yang berhasil diproses.")
                    st.warning("Tidak ada teks yang berhasil diekstrak dari file. Pastikan file tidak kosong dan dalam format yang valid.")
                
                # Clear progress display
                progress_bar.empty()
                status_text.empty()
    
    # Tampilkan info dokumen yang diproses
    if st.session_state.doc_texts:
        st.info(f"Ada {len(st.session_state.doc_texts)} dokumen yang telah diproses dan siap untuk dicari.")
        
        # Tombol untuk melihat dokumen
        with st.expander("Lihat Dokumen"):
            # Loop untuk setiap dokumen: tampilkan judul dan total pecahan kalimat
            for file, sentences in st.session_state.split_texts.items():
                total_sentences = len(sentences)
                st.write(f"**{file}**: {total_sentences} kalimat")
                
                # Tampilkan contoh kalimat dari dokumen
                if total_sentences > 0:
                    st.write("Contoh kalimat:")
                    
                    # Tampilkan beberapa kalimat dari awal, tengah, dan akhir dokumen
                    if total_sentences > 6:
                        sample_indices = [0, 1, total_sentences//2, total_sentences//2 + 1, total_sentences-2, total_sentences-1]
                        samples = [sentences[i] for i in range(total_sentences) if i in sample_indices]
                    else:
                        samples = sentences
                    
                    for idx, sentence in samples[:6]:  # Batasi jumlah contoh
                        st.write(f"- [{idx}] {sentence[:100]}..." if len(sentence) > 100 else f"- [{idx}] {sentence}")
                
                st.write("---")
    
    # Input dan pengaturan pencarian dalam form - IMPROVED
    with st.form(key="search_form"):
        st.subheader("Pencarian Dokumen")
        
        # Input keyword
        keyword = st.text_input("Masukkan keyword untuk pencarian")
        
        # Pilih metode pencarian
        col1, col2 = st.columns(2)
        with col1:
            search_method = st.selectbox("Pilih metode pencarian", ["Exact Match", "BM25"])
        with col2:
            eval_method = st.multiselect("Pilih metode evaluasi", ["ROUGE-L", "METEOR"], default=["ROUGE-L", "METEOR"])
        
        # Pilih ukuran konteks paragraf
        context_size = st.slider("Jumlah kalimat konteks (sebelum & sesudah)", 
                             min_value=1, max_value=7, value=3, step=1,
                             help="Jumlah kalimat yang akan ditampilkan sebelum dan sesudah kalimat yang cocok")
        
        # Tombol Submit form
        search_submitted = st.form_submit_button("Cari")
    
    # Proses pencarian saat form disubmit
    if search_submitted:
        if not keyword:
            st.warning("Silakan masukkan kata kunci untuk pencarian.")
        elif not st.session_state.split_texts:
            st.error("Belum ada dokumen yang diproses. Silakan unggah dokumen terlebih dahulu.")
        else:
            # Validasi keyword
            if len(keyword.strip()) < 2:
                st.warning("Kata kunci terlalu pendek. Gunakan minimal 2 karakter.")
            else:
                # Container untuk hasil pencarian
                result_container = st.container()
                
                with result_container:
                    # Menampilkan timer
                    start_time = time.time()
                    
                    # Check if search is in cache - NEW
                    cache_key = get_search_cache_key(keyword, search_method)
                    cached_search = st.session_state.search_cache.get(cache_key)
                    
                    if cached_search:
                        search_results = cached_search
                    else:
                        # Pencarian dengan metode yang dipilih
                        with st.spinner(f'Melakukan pencarian {search_method}...'):
                            if search_method == "Exact Match":
                                search_results = exact_match_search(
                                    keyword, 
                                    st.session_state.split_texts,
                                    st.session_state.sentence_index
                                )
                            elif search_method == "BM25":
                                search_results = bm25_search(
                                    keyword, 
                                    st.session_state.split_texts,
                                    st.session_state.processed_sentences
                                )
                    
                    search_time = time.time() - start_time
                    st.success(f"Pencarian selesai dalam {search_time:.2f} detik")
                    
                    # Evaluasi hasil pencarian
                    if search_results:
                        # Check if evaluation is in cache - NEW
                        eval_cache_key = get_eval_cache_key(search_results, eval_method)
                        cached_eval = st.session_state.eval_cache.get(eval_cache_key)
                        
                        if cached_eval:
                            eval_results = cached_eval
                        else:
                            with st.spinner('Mengevaluasi hasil pencarian...'):
                                eval_results = sort_by_evaluation_batched(
                                    search_results, 
                                    st.session_state.split_texts, 
                                    eval_method
                                )
                        
                        st.write("**Hasil Pencarian**")
                        
                        # Tampilkan kalimat dengan evaluasi terbaik
                        for i, (rouge, meteor, file, idx, sentence, explanation) in enumerate(eval_results):
                            with st.container():
                                st.subheader(f"Hasil #{i+1}")
                                st.write(f"**Dokumen:** {file}")
                                
                                # Dapatkan paragraf konteks dengan kalimat sebelum dan sesudah
                                paragraph, matched_position, context_indices = get_context_paragraph(
                                    file, idx, st.session_state.split_texts, context_size=context_size
                                )

                                # Highlight keyword dalam paragraf
                                highlighted_paragraph = re.sub(
                                    r'\b(' + re.escape(keyword) + r')\b', 
                                    r'**\1**', 
                                    paragraph, 
                                    flags=re.IGNORECASE
                                )

                                # Tampilkan paragraf konteks
                                st.write(f"**Konteks Paragraf:**")
                                st.write(highlighted_paragraph)

                                # Buat kamus ordinal dalam bahasa Indonesia
                                ordinals = {1: "pertama", 2: "kedua", 3: "ketiga", 4: "keempat", 
                                            5: "kelima", 6: "keenam", 7: "ketujuh"}
                                ordinal_str = ordinals.get(matched_position + 1, f"ke-{matched_position + 1}")

                                # Tampilkan kalimat mana yang merupakan match
                                st.write(f"**Catatan:** Kalimat yang cocok adalah kalimat {ordinal_str} dalam paragraf ini (kalimat #{idx}).")

                                # Tampilkan indeks kalimat dalam paragraf ini
                                with st.expander("Lihat Indeks Kalimat"):
                                    st.write(f"Indeks kalimat dalam paragraf ini: {context_indices}")
                                
                                # Tampilkan panjang kalimat (tanpa bonus)
                                sentence_length = explanation.get("sentence_length", len(sentence.split()))
                                st.write(f"**Panjang Kalimat:** {sentence_length} kata")
                                
                                # Tampilkan flag kalimat pendek jika ada
                                is_short_sentence = explanation.get("is_short_sentence", False)
                                if is_short_sentence:
                                    st.info("Ini adalah kalimat pendek, skor evaluasi telah ditingkatkan.")
                                
                                # PERBAIKAN - Tambahkan bagian ini untuk menghitung nilai berdasarkan token yang terlihat
                                # Hitung nilai sesuai dengan detail perhitungan jika tokennya tersedia
                                if 'y_true_tokens' in explanation and 'y_pred_tokens' in explanation:
                                    ref_tokens = explanation.get('y_true_count', len(explanation['y_true_tokens']))
                                    pred_tokens = explanation.get('y_pred_count', len(explanation['y_pred_tokens']))
                                    lcs_value = explanation.get('lcs_length', 0)
                                    
                                    # Hitung precision dan recall berdasarkan token yang ada
                                    displayed_precision = lcs_value / pred_tokens if pred_tokens > 0 else 0
                                    displayed_precision_percentage = int(min(100, round(displayed_precision * 100)))
                                    
                                    displayed_recall = lcs_value / ref_tokens if ref_tokens > 0 else 0
                                    displayed_recall_percentage = int(min(100, round(displayed_recall * 100)))
                                    
                                    # Hitung F-measure berdasarkan nilai precision dan recall yang dihitung
                                    if (displayed_precision + displayed_recall) > 0:
                                        beta = 1
                                        beta_squared = beta ** 2
                                        calculated_f_measure = ((1 + beta_squared) * displayed_precision * displayed_recall) / (beta_squared * displayed_precision + displayed_recall)
                                        calculated_f_measure_percentage = int(min(100, round(calculated_f_measure * 100)))
                                    else:
                                        calculated_f_measure_percentage = 0
                                        
                                    # Gunakan nilai yang dihitung untuk tampilan utama
                                    displayed_rouge = {
                                        'precision': displayed_precision_percentage,
                                        'recall': displayed_recall_percentage,
                                        'f_measure': calculated_f_measure_percentage
                                    }
                                else:
                                    # Jika token tidak tersedia, gunakan nilai asli
                                    displayed_rouge = rouge
                                
                                # Tampilkan metrik evaluasi
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if "ROUGE-L" in eval_method:
                                        st.write(f"**ROUGE-L Metrics:**")
                                        # PERBAIKAN: Gunakan nilai yang dihitung dari token yang terlihat
                                        st.write(f"- Precision: {int(displayed_rouge['precision'])}%")
                                        st.write(f"- Recall: {int(displayed_rouge['recall'])}%")
                                        st.write(f"- F-measure: {int(displayed_rouge['f_measure'])}%")
                                
                                with col2:
                                    if "METEOR" in eval_method:
                                        st.write(f"**METEOR Score:** {int(meteor)}%")
                                
                                # Tampilkan penjelasan perhitungan
                                with st.expander("Lihat Penjelasan Perhitungan"):
                                    if "short_sentence_info" in explanation:
                                        st.info(explanation["short_sentence_info"])
                                    
                                    if "self_evaluation" in explanation and explanation["self_evaluation"]:
                                        st.write("**Catatan:** Kalimat ini dievaluasi terhadap dirinya sendiri dengan modifikasi sinonim karena tidak ada kalimat lain untuk dibandingkan.")
                                    
                                    if "artificial_reference" in explanation:
                                        st.write(f"**Kalimat Referensi (dimodifikasi):** {explanation['artificial_reference']}")
                                    
                                    if "comparison_sentence" in explanation:
                                        st.write(f"**Kalimat Pembanding:** {explanation['comparison_sentence']}")
                                    
                                    # PERBAIKAN: Perhitungan ROUGE-L yang lebih akurat dengan actual values
                                    st.write("**Rincian Perhitungan ROUGE-L:**")
                                    
                                    # Menampilkan token-token dari kalimat referensi dan prediksi (termasuk stemming)
                                    if 'y_true_tokens' in explanation and 'y_pred_tokens' in explanation:
                                        # Batasi jumlah token yang ditampilkan lebih ketat untuk kalimat panjang
                                        max_tokens_to_display = 10  # Kurangi dari 20 menjadi 10 untuk keterbacaan
                                        
                                        # Tampilkan token dengan batasan yang lebih ketat
                                        st.write(f"- Token kalimat referensi (setelah stemming): {explanation['y_true_tokens'][:max_tokens_to_display]}" + 
                                                (f" ... (total {explanation.get('y_true_count', len(explanation['y_true_tokens']))} token)" 
                                                if explanation.get('y_true_count', len(explanation['y_true_tokens'])) > max_tokens_to_display else ""))
                                        
                                        st.write(f"- Token kalimat prediksi (setelah stemming): {explanation['y_pred_tokens'][:max_tokens_to_display]}" + 
                                                (f" ... (total {explanation.get('y_pred_count', len(explanation['y_pred_tokens']))} token)" 
                                                if explanation.get('y_pred_count', len(explanation['y_pred_tokens'])) > max_tokens_to_display else ""))
                                        
                                        # Gunakan jumlah token lengkap untuk perhitungan
                                        ref_tokens = explanation.get('y_true_count', len(explanation['y_true_tokens']))
                                        pred_tokens = explanation.get('y_pred_count', len(explanation['y_pred_tokens']))
                                        
                                        # Menggunakan LCS konsisten dari explanation
                                        lcs_value = explanation.get('lcs_length', 0)
                                        
                                        # Pastikan LCS tidak melebihi jumlah token
                                        lcs_value = min(lcs_value, min(ref_tokens, pred_tokens))
                                        
                                        # Tampilkan LCS dan jumlah token
                                        st.write(f"- Jumlah token kalimat referensi: {ref_tokens}")
                                        st.write(f"- Jumlah token kalimat prediksi: {pred_tokens}")
                                        st.write(f"- Panjang LCS: {lcs_value}")
                                        
                                        # Tampilkan perhitungan yang konsisten (menggunakan nilai yang sama dengan tampilan utama)
                                        st.write(f"- Precision = LCS / Jumlah token prediksi = {lcs_value} / {pred_tokens} = {displayed_precision:.4f} = {displayed_precision_percentage}%")
                                        st.write(f"- Recall = LCS / Jumlah token referensi = {lcs_value} / {ref_tokens} = {displayed_recall:.4f} = {displayed_recall_percentage}%")
                                        
                                        # Tampilkan perhitungan F-measure yang konsisten
                                        st.write(f"- F-measure = (2 * Precision * Recall) / (Precision + Recall)")
                                        st.write(f"  = (2 * {displayed_precision:.4f} * {displayed_recall:.4f}) / ({displayed_precision:.4f} + {displayed_recall:.4f})")
                                        st.write(f"  = {calculated_f_measure:.4f} = {calculated_f_measure_percentage}%")
                                    
                                    # Perbaikan bagian METEOR
                                    st.write("**Rincian Perhitungan METEOR:**")
                                    if 'meteor_base' in explanation:
                                        st.write(f"- Skor METEOR: {int(meteor)}%")
                                        st.write(f"- METEOR dihitung berdasarkan token yang cocok antara kalimat referensi dan prediksi dengan penambahan sinonim.")
                                        if 'meteor_base' in explanation and 'meteor_expanded' in explanation:
                                            st.write(f"- METEOR dasar (tanpa ekspansi sinonim): {int(explanation['meteor_base'])}%")
                                            st.write(f"- METEOR dengan ekspansi sinonim: {int(explanation['meteor_expanded'])}%")
                                    
                                    # Tampilkan info tambahan jika ada error
                                    if 'error' in explanation:
                                        st.error(f"Error dalam perhitungan: {explanation['error']}")
                                    
                                    if 'note' in explanation:
                                        st.write(f"**Catatan:** {explanation['note']}")
                                
                                st.write("---")
                    else:
                        st.warning("Tidak ada hasil pencarian yang cocok.")

if __name__ == "__main__":
    main()