# Document Q&A using Semantic Searcs

import numpy as np
from flask import Flask, render_template, request
from pymongo import MongoClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Your MongoDB link is kept in a separate private file: config.py
try:
    from config import MONGO_URI
except ImportError:
    raise SystemExit("config.py not found. Copy config_example.py to config.py and add your MongoDB link.")

# Settings (you can play with these numbers)
CHUNK_SIZE = 100   # words in one chunk
OVERLAP = 20       # words repeated between two chunks
TOP_K = 3          # how many answers to show
MIN_SCORE = 0.30   # below this, the answer is not relevant

# Connect to the database
client = MongoClient(MONGO_URI)
db = client["doc_qa_db"]
collection = db["chunks"]

# Load the AI model that converts text to numbers
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_chunks(text):
    """Cut a long text into chunks of about 100 words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = start + CHUNK_SIZE - OVERLAP
    return chunks


def read_file(file):
    """Read the text from a .pdf or .txt file."""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text = text + " " + page_text
        return text

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return None  # unsupported file type


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    pasted_text = request.form.get("text", "").strip()

    if file and file.filename != "":
        text = read_file(file)
        source = file.filename
        if text is None:
            return render_template("index.html", upload_message="Only PDF and TXT files are allowed.")
    elif pasted_text != "":
        text = pasted_text
        source = "pasted text"
    else:
        return render_template("index.html", upload_message="Please choose a file or paste some text.")

    chunks = split_into_chunks(text)
    if len(chunks) == 0 or text.strip() == "":
        return render_template("index.html", upload_message="No readable text found in this document.")

    embeddings = model.encode(chunks, normalize_embeddings=True)
    for i in range(len(chunks)):
        collection.insert_one({
            "text": chunks[i],
            "embedding": embeddings[i].tolist(),
            "source": source,
            "chunk_no": i + 1,
        })

    message = "Done! Saved " + str(len(chunks)) + " chunks from " + source
    return render_template("index.html", upload_message=message)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if question == "":
        return render_template("index.html", ask_message="Please type a question.")

    all_chunks = list(collection.find({}))
    if len(all_chunks) == 0:
        return render_template("index.html", question=question, ask_message="No documents uploaded yet.")

    question_vector = model.encode(question, normalize_embeddings=True)

    scored_chunks = []
    for chunk in all_chunks:
        score = float(np.dot(chunk["embedding"], question_vector))
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, chunk in scored_chunks[:TOP_K]:
        if score >= MIN_SCORE:
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": round(score, 2),
            })

    if len(results) == 0:
        return render_template("index.html", question=question, ask_message="No relevant answer found.")

    return render_template("index.html", question=question, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
