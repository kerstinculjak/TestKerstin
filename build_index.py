from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv
import os

# .env laden
load_dotenv(Path(__file__).parent / ".env")

# PDF-Pfad
pdf_path = Path(__file__).parent / "study_notes" / "ML3 Klassifikation.pdf"
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

# Text in kleine Stücke teilen
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks.")

# Embeddings erzeugen (numerische Repräsentationen)
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS-Index erstellen und speichern
db = FAISS.from_documents(chunks, embeddings)
index_path = Path(__file__).parent / "indexes"
index_path.mkdir(exist_ok=True)
db.save_local(str(index_path / "ml3_index"))
print("✅ Index gespeichert unter:", index_path / "ml3_index")
