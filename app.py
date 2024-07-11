import os
import streamlit as st
from typing import List, Tuple
from langchain.docstore.document import Document as LangchainDocument
import cohere
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.cohere import CohereEmbedding
from rag import *

# from dotenv import load_dotenv
# load_dotenv()
api_key=os.getenv('COHERE_KEY')

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Use the information contained in Antoine Bertin resumé to give a short answer to a question. 
        Respond only to the question asked, response should be short and relevant to the question.
        If Antoine Bertin resumé does not provide information about the question, respond only to book a video call with Antoine.
        The answer must always include some reference to Antoine's work as a data scientist from his resumé.
        """ 
    },
    {
        "role": "user", "content": 
        """resumé extract for context: {context}
---
Question: {question} """
    }
]

# Function to cache the LLM
@st.cache_resource
def get_client(prompt_in_chat_format=prompt_in_chat_format):
    """Returns a cached instance of the Open Source LLM."""
    READER_LLM = cohere.Client(api_key=api_key)
    RAG_PROMPT_TEMPLATE = format_prompt(prompt_in_chat_format)

    return READER_LLM, RAG_PROMPT_TEMPLATE

# Function to build and cache the vector db
@st.cache_resource
def load_and_cache_index(dir="index_cv"):
    """Loads and caches the faiss vdb index."""
    embed_model = CohereEmbedding(
        cohere_api_key=api_key,
        model_name="embed-english-v3.0",
    )
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    vdb = load_index_from_storage(storage_context, embed_model=embed_model)
    return vdb

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to reply to queries using the FAISS index
def answer_with_rag(question: str, llm, knowledge_index: VectorStoreIndex, prompt: str) -> Tuple[str, List[LangchainDocument]]:
    try:
        # Gather documents with retriever
        with st.spinner("reading Antoine's resumé..."):
            print("=> Retrieving documents...")
            top_k = 5 # how many documents to fetch on first pass
            top_n = 1 # how many documents to sub-select with rerank
            
            retriever = RetrieverWithRerank(knowledge_index.as_retriever(similarity_top_k=top_k), api_key=api_key)
            documents = retriever.retrieve(question, top_n=top_n)

            # Build the final prompt
            context = "\nExtracted text from Antoine Bertin's resumé:\n"
            context += documents[0]['text']
            final_prompt = prompt.format(question=question, context=context)

        # Redact an answer
        with st.spinner("Processing your query... be cool, this runs on free resources! 😅"):
            print("=> Generating answer...")
            print(type(final_prompt))
            resp = llm.chat(message=final_prompt, model="command-r", temperature=0.)
            answer = resp.text
            add_message(answer)

    except Exception as e:
        st.error(f"Error during query processing: {e}")
        print(f"Debug: {e}")

# Main application logic
def main():
    """Main function to run the application logic."""
    st.warning("⚠️ This chatbot uses a small free open source LLM and run on a free server instance 😅, so please be patient...🙏")
    st.info("🎉 Vector DB and LLM are cached after their first run | Oh and LLM is english only 🙏")
    
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []
    # LinkedIn button-like link with space, transparent background, and emoji
    st.sidebar.markdown("""
        <a href="https://www.linkedin.com/in/antoinebertin35/" target="_blank" style="text-decoration: none;">
            <button style="margin-bottom: 10px; color: #0A66C2; background-color: transparent; border: 2px solid #0A66C2; border-radius: 5px; padding: 10px; cursor: pointer;">
                👤 My LinkedIn
            </button>
        </a>
        """, unsafe_allow_html=True)

    # Calendly button-like link with transparent background and emoji
    st.sidebar.markdown("""
        <a href="https://calendly.com/antoinebertin/30/" target="_blank" style="text-decoration: none;">
            <button style="color: #4CAF50; background-color: transparent; border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; cursor: pointer;">
                📅 My Calendly
            </button>
        </a>
        """, unsafe_allow_html=True)
    # Define the list of tasks
    tasks = [
        "Scaling Up Server Instance",
        "Add memory",
        "Prompt injection",
        "Implementing a Reranker for RAG",
        "Fine-Tuning the LLM for Task Focus",
        "Packaging the LLM for Low Latency: Quantize and deploy the model on edge to reduce latency"
    ]

    # Create a bulleted list in the sidebar
    st.sidebar.write("## What's next?")
    for task in tasks:
        st.sidebar.markdown(f"- {task}")
    
    try:
        # Check if vdb is already loaded
        if 'vdb' not in st.session_state:
            with st.spinner("loading the vector database..."):
                print("=> VDB getting ready...")
                st.session_state['vdb'] = load_and_cache_index()
        else:
            print("=> VDB already loaded.")
            vdb = st.session_state['vdb']

        # Check if llm is already loaded
        if 'llm' not in st.session_state or 'RAG_PROMPT_TEMPLATE' not in st.session_state:
            with st.spinner("loading the free open source LLM..."):
                print("=> getting client LLM...")
                llm, RAG_PROMPT_TEMPLATE = get_client()
                st.session_state['llm'] = llm
                st.session_state['RAG_PROMPT_TEMPLATE'] = RAG_PROMPT_TEMPLATE
        else:
            print("=> LLM already loaded.")
            llm = st.session_state['llm']
            RAG_PROMPT_TEMPLATE = st.session_state['RAG_PROMPT_TEMPLATE']
    except Exception as e:
        st.error(f"Error loading index: {e}")
        print(f"Debug: {e}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["agent"]):
            st.write(message["content"])

    query = st.chat_input("Chat with Antoine Bertin's resume")

    if not st.session_state.messages:
        add_message("Hi! Want to learn more about Antoine's career and skills? Let's get started!")

    if query:
        add_message(query, agent="human", stream=False, store=True)
        answer_with_rag(query, llm, vdb, RAG_PROMPT_TEMPLATE)

if __name__ == "__main__":
    main()