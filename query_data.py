import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from create_database import EMBEDDINGS, LLM

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Ответь на вопрос, основываясь только на следующем контексте:

{context}

---

Ответь на вопрос, основываясь на указанном выше контексте: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = EMBEDDINGS
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    for doc, score in results:
        print(f"Оценка: {score:.3f} | Текст: {doc.page_content[:100]}...")
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Совпадения не найдены.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = LLM
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Ответ: {response_text}\nРесурс: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()