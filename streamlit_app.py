import streamlit as st
from openai import OpenAI
import os
import docloader
import chat_openrouter
import embedder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    layout="wide", page_title="Root game helper - OpenRouter chatbot app")
st.title("Root game helper ")

UPLOAD_FOLDER = "data/uploaded_pdfs"
# UPLOAD_FOLDER = "data/pdf"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template = """
You are a chatbot that helps with the rules of the boardgame Root. Use English language by default.
If you don't know the anwser, just say that you don't know.
Question: {question}
Context: {context}
Anwser:
"""


hello_msg = """
Hi I am a Root assistant!
Root is a strategic and thematic board game for 1 to 6 players, where participants compete to become the most powerful entity controlling the vast Woodland. Here's a summary based on the context provided:

Core Mechanics:
Map: Gameplay revolves around a map depicting distinct "clearings" connected by paths and containing forests.

Factions: Players choose from one of four unique factions (Marquise de Cat, Eyrie Dynasties, Forest Alliance, Riverfolk Company) who approach the Woodland in distinct ways.

Woodland Control: Players use their "warriors" to rule clearings. A ruling faction can perform actions like moving warriors, potentially building "victory point" buildings specific to their faction.

Hirelings: Earn special "hireling cards" used each turn for specific actions, controlled by "control markers" placed on them.

Flow: Turns consist of three phases: Birdsong, Daylight, and Evening, with specific actions dictated by both the turn phase and the chosen faction.

Victory: Winning requires achieving 30 "victory points" through faction-specific methods (e.g., building structures) or completing special "dominance" cards.

The game emphasizes organic learning, with mechanics flowing through the interaction on the map and via the turn phases defined by faction cards.

"""

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


with st.sidebar:
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": hello_msg}]
        st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": hello_msg}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Default fallback message
    response_content = "I'm sorry, an error occurred or no relevant information could be found."

    if st.session_state.faiss_index is not None:
        k_relevant_docs = 3
        retrieved_docs = embedder.retrieve_docs(
            prompt, st.session_state.faiss_index, k=k_relevant_docs)

        if retrieved_docs:
            response_obj = anwser_question(prompt, retrieved_docs, model)
            response_content = response_obj.content
        else:
            st.warning(
                "No relevant document chunks found for your query. Answering without specific context.")
            prompt_no_context = ChatPromptTemplate.from_template(
                "Question: {question}\nAnswer:")
            chain_no_context = prompt_no_context | model
            response_obj = chain_no_context.invoke({"question": prompt})
            response_content = response_obj.content
    else:
        st.warning(
            "No documents have been loaded and indexed. Please ensure files are in 'data/uploaded_pdfs' when the app starts.")
        prompt_no_context = ChatPromptTemplate.from_template(
            "Question: {question}\nAnswer:")
        chain_no_context = prompt_no_context | model
        response_obj = chain_no_context.invoke({"question": prompt})
        response_content = response_obj.content

    st.session_state.messages.append(
        {"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)

# if prompt := st.chat_input():
#     if not api_key:
#         st.info("Invalid API key.")
#         st.stop()
#     client = OpenAI(api_key=api_key, base_url=base_url)
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     if "faiss_index" in st.session_state and st.session_state.faiss_index is not None:
#         # 1. Retrieve relevant documents based on the prompt
#         retrieved_docs = embedder.retrieve_docs(
#             prompt, st.session_state.faiss_index, k=3)
#         # 2. Pass *only* the retrieved documents to the answering function
#         response = anwser_question(prompt, retrieved_docs, model)
#     else:
#         # Fallback if no documents are indexed, just answer with the LLM without context
#         # Or handle this case by informing the user to upload documents
#         st.warning(
#             "Please upload and process documents first for context-aware answers.")
#         # You might create a simpler chain here without context
#     prompt_no_context = ChatPromptTemplate.from_template(
#         "Question: {question}\nAnswer:")
#     chain_no_context = prompt_no_context | model
#     response = chain_no_context.invoke({"question": prompt})
#     response = anwser_question(prompt, documents, model)
#     # response = client.chat.completions.create(
#     #     model=selected_model,
#     #     messages=st.session_state.messages
#     # )
#     msg = response.content
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)
