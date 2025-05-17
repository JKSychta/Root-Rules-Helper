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

UPLOAD_FOLDER = "data/uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template = """
You are an assistant for question-anwsering tasks. Use Polish language by default.
If you don't know the nawser, just sa that you don't know. Use three sentences maximum anwsers.
Question: {question}
Context: {context}
Anwser:
"""

# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "google/gemma-3-1b-it:free"
model = chat_openrouter.ChatOpenRouter(model_name=selected_model)


def anwser_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


with st.sidebar:
    uploaded_files = st.file_uploader(
        label="Please insert a text file.", accept_multiple_files=True)


if uploaded_files:
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    # Now you can confidently loop over the list of UploadedFile objects
    for uploaded_file_obj in uploaded_files:  # Renamed for clarity
        # Check if the object is not None (shouldn't happen with file_uploader typically, but good practice)
        if uploaded_file_obj is not None:
            file_name = uploaded_file_obj.name
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file_obj.getbuffer())

            st.write(f"File '{file_name}' uploaded successfully!")
    try:
        documents = docloader.load_documents_from_folder(UPLOAD_FOLDER)
        if documents:
            st.session_state.faiss_index = embedder.create_index(documents)
            st.write("All uploaded documents processed and indexed successfully!")
            # Consider a better name like st.session_state.documents_indexed
            st.session_state.retrive_files = True
        else:
            st.warning("No documents found in the upload folder to process.")

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        # Optionally reset state if processing failed
        st.session_state.faiss_index = None
        st.session_state.retrive_files = False

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
    response = anwser_question(prompt, documents, model)
    # response = client.chat.completions.create(
    #     model=selected_model,
    #     messages=st.session_state.messages
    # )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
