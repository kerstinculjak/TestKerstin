import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env laden
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# LLM initialisieren
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.4)),
    max_retries=8,
    timeout=60,
)

def ask(question: str) -> str: #Gibt den Antworttext zurück
    return llm.invoke(question).content

def main():
    # 1) Einmalige Frage via Argument: python3 llm_client.py "Deine Frage"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print("Frage:", q)
        print("Antwort:", ask(q))
        return

    # 2) Interaktiver Modus
    print("Interaktiver Modus. Tippe deine Frage und drücke Enter.")
    print("Beenden mit: exit / quit / /exit\n")
    while True:
        try:
            q = input("Frage> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "/exit"}:
                print("Tschüss! 👋")
                break
            print("\nAntwort:", ask(q), "\n")
        except KeyboardInterrupt:
            print("\nAbgebrochen. 👋")
            break

if __name__ == "__main__":
    main()


