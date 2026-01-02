#!/usr/bin/env python3
"""
RAG backend (TF-IDF + BM25 hybrid retriever) + optional OpenAI LLM.
Automatically removes any source labels from documents to prevent LLM outputting sources.
"""

import os, re, pickle, json
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# NLTK sentence tokenizer
_USE_NLTK = False
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    from nltk.tokenize import sent_tokenize
    _USE_NLTK = True
except Exception:
    try:
        nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize
        _USE_NLTK = True
    except Exception:
        _USE_NLTK = False

def safe_sent_tokenize(text: str) -> List[str]:
    if _USE_NLTK:
        return sent_tokenize(text)
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]

# ---------------------- CONFIG ----------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./docs"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./rag_store"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_FILE = MODEL_DIR / "tfidf.pkl"
BM25_FILE = MODEL_DIR / "bm25.pkl"
MAPPING_FILE = MODEL_DIR / "mapping.pkl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
WEIGHT_TFIDF = 0.5
WEIGHT_BM25 = 0.5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------- HELPERS ----------------------

def clean_text(text: str) -> str:
    """
    Remove any lines containing 'Source:', file paths, chunk IDs, or similar.
    """
    # remove lines starting with "Source:"
    text = re.sub(r"(?im)^source:.*$", "", text)
    # remove any remaining chunk ID labels like "#cid=123"
    text = re.sub(r"#cid=\d+", "", text)
    return text.strip()

def load_text_files(folder: Path) -> List[Tuple[str, str]]:
    folder.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except:
                text = path.read_text(encoding="latin-1", errors="ignore")
            text = clean_text(text)  # <-- clean here
            if text.strip():
                docs.append((str(path), text))
    return docs

def chunk_text(text: str) -> List[str]:
    sents = safe_sent_tokenize(text)
    if not sents: sents = [p for p in text.splitlines() if p.strip()]
    chunks, cur = [], ""
    for s in sents:
        if len(cur)+len(s)+1 <= CHUNK_SIZE:
            cur = (cur+" "+s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    if CHUNK_OVERLAP>0 and len(chunks)>1:
        new_chunks = []
        for i, c in enumerate(chunks):
            if i==0: new_chunks.append(c)
            else: new_chunks.append((new_chunks[-1][-CHUNK_OVERLAP:]+" "+c).strip())
        return new_chunks
    return chunks

def build_indexes(docs: List[Tuple[str,str]]):
    mapping, chunk_texts = [], []
    for path, text in docs:
        for i, c in enumerate(chunk_text(text)):
            mapping.append({"path": path, "chunk_id": i})
            chunk_texts.append(c)
    if not chunk_texts: raise ValueError("No text to index.")
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(chunk_texts)
    tfidf_norm = normalize(tfidf_matrix)
    if BM25Okapi:
        tokenized = [re.findall(r"\w+", t.lower()) for t in chunk_texts]
        bm25 = BM25Okapi(tokenized)
    else:
        bm25 = None
    with open(TFIDF_FILE, "wb") as f: pickle.dump({"vectorizer":tfidf,"matrix":tfidf_norm},f)
    with open(MAPPING_FILE,"wb") as f: pickle.dump({"mapping":mapping,"texts":chunk_texts},f)
    if bm25: 
        with open(BM25_FILE,"wb") as f: pickle.dump(bm25,f)

def load_indexes():
    if not TFIDF_FILE.exists() or not MAPPING_FILE.exists(): return None
    with open(TFIDF_FILE,"rb") as f: tfidf_store = pickle.load(f)
    with open(MAPPING_FILE,"rb") as f: mapping_store = pickle.load(f)
    bm25 = None
    if BM25_FILE.exists(): 
        with open(BM25_FILE,"rb") as f: bm25 = pickle.load(f)
    return tfidf_store, mapping_store, bm25

def hybrid_retrieve(query: str, k:int=TOP_K) -> List[str]:
    loaded = load_indexes()
    if not loaded: return []
    tfidf_store, mapping_store, bm25 = loaded
    texts = mapping_store["texts"]
    vectorizer = tfidf_store["vectorizer"]
    matrix = tfidf_store["matrix"]
    q_vec = normalize(vectorizer.transform([query]))
    tfidf_scores = (matrix @ q_vec.T).toarray().ravel()
    if bm25:
        tokenized_q = re.findall(r"\w+", query.lower())
        bm25_scores = np.array(bm25.get_scores(tokenized_q))
    else: bm25_scores = np.zeros_like(tfidf_scores)
    def safe_normalize(arr):
        if arr.size==0: return arr
        if arr.max()==arr.min(): return np.ones_like(arr)*0.5
        return (arr-arr.min())/(arr.max()-arr.min())
    combined = WEIGHT_TFIDF*safe_normalize(tfidf_scores)+WEIGHT_BM25*safe_normalize(bm25_scores)
    top_idx = np.argsort(combined)[::-1][:k]
    # RETURN ONLY CLEAN TEXT
    return [re.sub(r"(?i)source:.*","",texts[i]).strip() for i in top_idx]

def assemble_prompt(query: str, retrieved_texts: List[str], max_context_chars: int=2000) -> str:
    if not retrieved_texts:
        return f"You are a helpful assistant.\nNo context available.\nQUESTION: {query}\nAnswer naturally. Do NOT mention sources."
    ctx = "\n\n---\n\n".join(retrieved_texts)[:max_context_chars]
    return f"You are a helpful assistant. Use the context below to answer the question.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\nAnswer naturally. Do NOT mention sources, paths, or chunk IDs."

# ---------------------- LLM ----------------------
def call_llm_openai(prompt: str) -> str:
    import requests
    if not OPENAI_API_KEY: return "[OpenAI API key missing]"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization":f"Bearer {OPENAI_API_KEY}","Content-Type":"application/json"}
    body = {"model":OPENAI_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":512,"temperature":0}
    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    try: return r.json()["choices"][0]["message"]["content"]
    except: return json.dumps(r.json())

def call_llm(prompt: str) -> str:
    try: return call_llm_openai(prompt)
    except Exception as e: return f"[LLM error: {e}]\nPrompt:\n{prompt[:500]}"

# ---------------------- FastAPI ----------------------
from fastapi import FastAPI

app = FastAPI(title="RAG TF-IDF+BM25")

class QueryIn(BaseModel):
    query: str
    top_k: int = TOP_K

@app.get("/")
async def root(): return {"ok": True, "message":"RAG TF-IDF+BM25 demo"}

@app.post("/ingest")
async def ingest_files():
    docs = load_text_files(DATA_DIR)
    if not docs: return {"ok":False,"message":"No documents found."}
    try: build_indexes(docs)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    return {"ok":True,"num_docs":len(docs)}

@app.post("/upload")
async def upload(file: UploadFile=File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    fname = DATA_DIR / file.filename
    fname.write_bytes(content)
    docs = load_text_files(DATA_DIR)
    if not docs: raise HTTPException(status_code=400, detail="No valid docs")
    build_indexes(docs)
    return {"ok":True,"filename":str(fname)}

@app.post("/query")
async def query_endpoint(inq: QueryIn):
    if not TFIDF_FILE.exists(): raise HTTPException(status_code=400, detail="Indexes not found")
    retrieved_texts = hybrid_retrieve(inq.query, k=inq.top_k)
    prompt = assemble_prompt(inq.query, retrieved_texts)
    answer = call_llm(prompt)
    return {"answer": answer}

@app.get("/health")
async def health(): return {"ok": True}

# ---------------------- CLI ----------------------
if __name__=="__main__":
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest",action="store_true")
    parser.add_argument("--query",type=str,default=None)
    parser.add_argument("--serve",action="store_true")
    args = parser.parse_args()
    if args.ingest:
        docs = load_text_files(DATA_DIR)
        if not docs: print("No docs found.")
        else:
            build_indexes(docs)
            print("Indexes built.")
    elif args.query:
        retrieved_texts = hybrid_retrieve(args.query)
        prompt = assemble_prompt(args.query,retrieved_texts)
        print(call_llm(prompt))
    elif args.serve:
        uvicorn.run(app,host="0.0.0.0",port=8000)
        