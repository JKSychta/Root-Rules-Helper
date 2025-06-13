# import os
# import fitz


# def load_pdf(file_path):
#     doc = fitz.open(file_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     doc.close()
#     return text


# def load_documents_from_folder(folder_path):
#     documents = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             text = load_pdf(os.path.join(folder_path, filename))
#             documents.append({"filename": filename, "text": text})
#     return documents
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

    # Initialize your text splitter
    # Experiment with chunk_size and chunk_overlap!
    # A chunk_size of 500-1000 tokens is a good starting point for LLMs.
    # 1 token is roughly 4 characters in English, so 500 tokens ~ 2000 characters.
    # chunk_overlap ensures context isn't lost at chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Example: 1000 characters (adjust for tokens)
        chunk_overlap=200,  # Example: 200 characters overlap
        length_function=len,  # Use character length
        add_start_index=True,  # Add start index to metadata for debugging
    )

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_text = load_pdf(os.path.join(folder_path, filename))

            # Create a LangChain Document object from the full text
            # This allows the splitter to add metadata
            langchain_document = Document(
                page_content=full_text, metadata={"filename": filename})

            # Split the document into chunks
            chunks = text_splitter.split_documents([langchain_document])

            # Each chunk is now a Document object with page_content and metadata
            # You can extract text and filename from the chunk's page_content and metadata
            for i, chunk in enumerate(chunks):
                # Optionally add chunk specific metadata
                chunk.metadata["chunk_index"] = i
                all_chunks.append({
                    "text": chunk.page_content,
                    "filename": chunk.metadata["filename"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    # ... add other useful metadata from chunk.metadata if needed
                })
    return all_chunks  # Return a list of dictionaries, where each dict is a chunk
