import streamlit as st
from openai import OpenAI
import os
import docloader
import chat_openrouter
import embedder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

# UPLOAD_FOLDER = "data/uploaded_pdfs"
UPLOAD_FOLDER = "data/pdf"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template = """
You are a board game rulebook helper. Use English language by default.
If you don't know the anwser, just say that you don't know.
Question: {question}
Context: {context}
Anwser:
"""

# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
# selected_model = "google/gemma-3-1b-it:free"
# <- wybrano model deepseek/deepseek-r1-0528-qwen3-8b:free bo miał największą ilość darmowych tokenów co jest potrzebne przy ilośći tekstu wymagenaj przez instrukcje
selected_model = st.secrets["MODEL"]
model = chat_openrouter.ChatOpenRouter(model_name=selected_model)


def anwser_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# --- START OF MODIFIED SECTION FOR INITIAL LOADING ---

# Initialize session state variables if they don't exist
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "documents_indexed" not in st.session_state:
    # A flag to know if indexing has occurred
    st.session_state.documents_indexed = False

# Define a function to load and index documents (can be called once)


@st.cache_resource  # Use st.cache_resource to cache heavy objects like FAISS index
def load_and_index_documents(folder_path):
    try:
        documents = docloader.load_documents_from_folder(folder_path)
        if documents:
            faiss_index = embedder.create_index(documents)
            st.success(
                "All pre-existing documents processed and indexed successfully!")
            return faiss_index
        else:
            st.info("No pre-existing documents found in the initial folder.")
            return None
    except Exception as e:
        st.error(f"Error processing pre-existing documents: {e}")
        return None


# Load and index documents at startup, but only if not already done
if not st.session_state.documents_indexed:
    st.session_state.faiss_index = load_and_index_documents(UPLOAD_FOLDER)
    if st.session_state.faiss_index is not None:
        st.session_state.documents_indexed = True

# --- END OF MODIFIED SECTION FOR INITIAL LOADING ---
# with st.sidebar:
#     uploaded_files = st.file_uploader(
#         label="Please insert a text file.", accept_multiple_files=True)


# if uploaded_files:
#     if not isinstance(uploaded_files, list):
#         uploaded_files = [uploaded_files]

#     # Now you can confidently loop over the list of UploadedFile objects
#     for uploaded_file_obj in uploaded_files:  # Renamed for clarity
#         # Check if the object is not None (shouldn't happen with file_uploader typically, but good practice)
#         if uploaded_file_obj is not None:
#             file_name = uploaded_file_obj.name
#             file_path = os.path.join(UPLOAD_FOLDER, file_name)
#             os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file_obj.getbuffer())

#             st.write(f"File '{file_name}' uploaded successfully!")
#     try:
#         documents = docloader.load_documents_from_folder(UPLOAD_FOLDER)
#         if documents:
#             st.session_state.faiss_index = embedder.create_index(documents)
#             st.write("All uploaded documents processed and indexed successfully!")
#             # Consider a better name like st.session_state.documents_indexed
#             st.session_state.retrive_files = True
#         else:
#             st.warning("No documents found in the upload folder to process.")

#     except Exception as e:
#         st.error(f"Error processing documents: {e}")
#         # Optionally reset state if processing failed
#         st.session_state.faiss_index = None
#         st.session_state.retrive_files = False

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if "faiss_index" in st.session_state and st.session_state.faiss_index is not None:
        # 1. Retrieve relevant documents based on the prompt
        retrieved_docs = embedder.retrieve_docs(
            prompt, st.session_state.faiss_index, k=3)
        # 2. Pass *only* the retrieved documents to the answering function
        response = anwser_question(prompt, retrieved_docs, model)
    else:
        # Fallback if no documents are indexed, just answer with the LLM without context
        # Or handle this case by informing the user to upload documents
        st.warning(
            "Please upload and process documents first for context-aware answers.")
        # You might create a simpler chain here without context
    prompt_no_context = ChatPromptTemplate.from_template(
        "Question: {question}\nAnswer:")
    chain_no_context = prompt_no_context | model
    response = chain_no_context.invoke({"question": prompt})
    response = anwser_question(prompt, documents, model)
    # response = client.chat.completions.create(
    #     model=selected_model,
    #     messages=st.session_state.messages
    # )
    msg = response.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
