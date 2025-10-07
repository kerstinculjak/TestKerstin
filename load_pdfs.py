from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# Pfad zu deinem PDF im Unterordner "study_notes"
pdf_path = Path(__file__).parent / "study_notes" / "ML3 Klassifikation.pdf"

# PDF laden
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

print(f"Loaded {len(pages)} pages from '{pdf_path.name}'")
