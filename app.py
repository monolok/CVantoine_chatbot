import os
import streamlit as st
import time
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
#from transformers import AutoTokenizer
from typing import List, Tuple
#from transformers import Pipeline
#from transformers import pipeline
#from transformers import AutoModelForCausalLM
from langchain.docstore.document import Document as LangchainDocument

# from dotenv import load_dotenv
# load_dotenv()

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Use the information contained in Antoine Bertin resumé to give a short answer to a question. 
        Respond only to the question asked, response should be short and relevant to the question.
        If Antoine Bertin resumé does not provide information about a question, respond only to book a video call with Antoine.
        """ 
    },
    {
        "role": "user", "content": 
        """Antoine Bertin resumé: {context}
---
Question: {question} """
    }
]

def format_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"System: {message['content']}\n"
        elif message["role"] == "user":
            prompt += f"User: {message['content']}\n"
    return prompt

# Function to cach the LLM
@st.cache_resource
def get_client(prompt_in_chat_format=prompt_in_chat_format, READER_MODEL_NAME=READER_MODEL_NAME):
    """Returns a cached instance of the Open Source LLM."""
    #model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)#, quantization_config=bnb_config)
    #tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    # READER_LLM = pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task="text-generation",
    #     do_sample=True,
    #     temperature=0.2,
    #     repetition_penalty=1.1,
    #     return_full_text=False,
    #     max_new_tokens=150,
    # )
    hf_key = os.getenv('HF_KEY')
    READER_LLM = InferenceClient(token=hf_key, model=READER_MODEL_NAME)
    #RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)
    RAG_PROMPT_TEMPLATE = format_prompt(prompt_in_chat_format)

    return READER_LLM, RAG_PROMPT_TEMPLATE

# Function to build and cache the vector db
@st.cache_resource
def load_and_cache_index(dir="open_source_model_vdb"):
    """Loads and caches the faiss vdb index."""
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    vdb = FAISS.load_local(dir, embeddings, allow_dangerous_deserialization=True) #  take the same model used for the embeddings
    return vdb

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to add a message to the chat
def add_message(msg, agent="ai", stream=True, store=True):
    """Adds a message to the chat interface, optionally streaming the output."""
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))

# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)

# Function to reply to queries using the FAISS index
def answer_with_rag(question: str, llm, knowledge_index: FAISS, prompt: str, num_retrieved_docs: int = 4) -> Tuple[str, List[LangchainDocument]]:
    try:
        # Gather documents with retriever
        with st.spinner("reading Antoine's resumé..."):
            print("=> Retrieving documents...")
            relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
            relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

            # Build the final prompt
            context = "\nExtracted text from Antoine Bertin's resumé:\n"
            context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

            final_prompt = prompt.format(question=question, context=context)

        # Redact an answer
        with st.spinner("Processing your query... be cool, this runs on free resources! 😅"):
            print("=> Generating answer...")
            #answer = llm(final_prompt)[0]["generated_text"]
            answer = llm.text_generation(prompt=final_prompt, temperature=0.2, do_sample=True, repetition_penalty=1.1, return_full_text=False, max_new_tokens=150)
            #Create answer with extract
            context = context.replace("Document 0:::", "\nDocument 0:::")
            context = context.replace("Document 1:::", "\n\nDocument 1:::")
            context = context.replace("Document 2:::", "\n\nDocument 2:::")
            context = context.replace("Document 3:::", "\n\nDocument 3:::")
            #answer += "\n\n---\n" + context
            print("------------------------------------------------------------------------")
            print(answer)
            print("------------------------------------------------------------------------")
            print(context)
            #add_message(stream_response(answer))
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