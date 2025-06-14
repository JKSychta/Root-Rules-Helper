
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # Import Document class


def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_documents_from_folder(folder_path):
    all_chunks = []  # We'll store chunks here


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Example: 1000 characters (adjust for tokens)
        chunk_overlap=100,  # Example: 200 characters overlap
        length_function=len,  # Use character length
        add_start_index=True,  # Add start index to metadata for debugging
    )

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_text = load_pdf(os.path.join(folder_path, filename))

            langchain_document = Document(
                page_content=full_text, metadata={"filename": filename})

            chunks = text_splitter.split_documents([langchain_document])

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                all_chunks.append({
                    "text": chunk.page_content,
                    "filename": chunk.metadata["filename"],
                    "chunk_index": chunk.metadata["chunk_index"],
                })
    return all_chunks
