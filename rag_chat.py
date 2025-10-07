from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
import os

# .env laden
load_dotenv(Path(__file__).parent / ".env")

# Index laden
index_path = Path(__file__).parent / "indexes" / "ml3_index"
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Chatmodell
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

# Prompt-Template
prompt = ChatPromptTemplate.from_template(
    "Nutze den folgenden Kontext, um die Frage zu beantworten.\n"
    "Wenn die Antwort nicht im Kontext steht, sag 'Das steht nicht in deinen Unterlagen.'\n\n"
    "Kontext:\n{context}\n\nFrage: {question}"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interaktive Eingabe
print("📚 RAG-Chat: Fragen zu ML3 Klassifikation stellen (oder 'exit' eingeben).")
while True:
    q = input("Frage> ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    answer = rag_chain.invoke(q)
    print("\nAntwort:\n", answer, "\n")
