import os
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



CHROMA_PATH = "chroma"
DATA_PATH = os.path.join("data", "accessory_exercises")

OLLAMA_URL = "http://localhost:11434"

LLM = ChatOllama(
    model="qwen3:8b",
    base_url=OLLAMA_URL,
    temperature=0.7,
)

EMBEDDINGS = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL,
)


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents() -> list[Document]:
    """
    Загружает все .md файлы из указанной директории.
    Возвращает список объектов Document, совместимых с LangChain.
    """

    documents = []
    data_directory = Path(DATA_PATH)

    if not data_directory.exists():
        print(f"Предупреждение: Директория '{DATA_PATH}' не найдена.")
        return []

    for file_path in data_directory.glob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():  # Проверяем, что файл не пустой
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path)}
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Документ {len(documents)} был разделен на {len(chunks)} частей.")

    document = chunks[10]
    print(f"{document.page_content}\n{document.metadata}")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Очищаем базу, если она есть
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Создаём новую базу
    db = Chroma.from_documents(
        chunks,
        EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )
    print(f"Часть {len(chunks)} была сохранена в {CHROMA_PATH}.")


if __name__ == "__main__":
    main()