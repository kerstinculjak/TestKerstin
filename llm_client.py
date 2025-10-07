import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.4)),
)

response = llm.invoke("Sag mir einen lustigen Fakt über Katzen.")
print(response.content)

