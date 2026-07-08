import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATASET_PATH = "dataset"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_PATTERN = re.compile(
    r"={20,}\n(?P<section_heading>.*?)\n={20,}\n\n(?P<section_body>.*?)(?=\n={20,}\n.*?\n={20,}\n\n|\n={20,}\nEND OF DOCUMENT\n={20,}\n?$)",
    re.DOTALL,
)


def extract_metadata_and_sections(doc):
    text = doc.page_content
    source = doc.metadata.get("source", "Unknown source")

    title_match = re.search(r"Title:\s*(.*?)\n", text)
    description_match = re.search(r"Description:\s*(.*?)\n\n={20,}", text, re.DOTALL)

    title = title_match.group(1).strip() if title_match else "Unknown title"
    description = description_match.group(1).strip() if description_match else ""

    sections = []
    for match in SECTION_PATTERN.finditer(text):
        section_heading = match.group("section_heading").strip()
        section_body = match.group("section_body").strip()
        section_id_match = re.match(r"^(\d+(?:\.\d+)*)", section_heading)
        section_id = section_id_match.group(1) if section_id_match else "unknown"

        cleaned_content = (
            f"Title: {title}\n\n"
            f"Section: {section_heading}\n\n"
            f"{section_body}"
        )

        sections.append(
            Document(
                page_content=cleaned_content,
                metadata={
                    "source": source,
                    "title": title,
                    "description": description,
                    "section_id": section_id,
                    "section_heading": section_heading,
                },
            )
        )

    return sections


def load_section_chunks():
    documents = []
    section_chunks = []

    for file in sorted(os.listdir(DATASET_PATH)):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(DATASET_PATH, file)
        docs = TextLoader(file_path, encoding="utf-8").load()
        documents.extend(docs)

        for doc in docs:
            section_chunks.extend(extract_metadata_and_sections(doc))

    return documents, section_chunks


def main():
    documents, chunks = load_section_chunks()

    # Use section-aware policy chunks instead of generic fixed-size splits.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} section-aware chunks.")
    print(f"Saved FAISS index to: {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    main()