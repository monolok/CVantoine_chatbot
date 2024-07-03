import streamlit as st
import time
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from typing import List, Tuple
from transformers import Pipeline
from transformers import pipeline
from transformers import AutoModelForCausalLM
from langchain.docstore.document import Document as LangchainDocument

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

#! FIX PROMPT SEE ANSWER BELOW:
# Based on the provided context, there is no direct information to determine whether the individual mentioned in 
# Document 1 is very good at coding. The statement "In 2024, Jedha was awarded the title of best coding BootCamp by Course Report" 
# suggests that Jedha, as a whole, received this recognition, but it does not necessarily imply that all instructors or students at Jedha 
# are exceptional coders. However, the individual's role as an assistant to students with coding exercises and case studies 
# could indicate some proficiency in coding. Without further context, it is unclear if this person is specifically known for their 
# coding abilities. If more information is needed to confirm their expertise in coding, it would be suggested to schedule a video 
# call with Antoine Bertin (mentioned in the initial prompt) or explore the resources available through the 'antoine' package.

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context, give a comprehensive answer to the question. 
        Respond only to the question asked, response should be short and relevant to the question. 
        Do not respond anything that can't be answered from the context.""" 
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
If the following question cannot be answered from the context suggest the following:
- Schedule a video call with Antoine Bertin for more detailed information. You can do this by visiting his Calendly link: 'https://calendly.com/antoinebertin/30'.
- Learn more about Antoine's story, you can download the 'antoine' package. To do this, use the command 'pip install antoine' in your terminal and run the 'hire_antoine' command.
   
Now here is the question you need to answer.

Question: {question}"""
    }
]

# Function to cach the LLM
@st.cache_resource
def get_client(prompt_in_chat_format=prompt_in_chat_format, READER_MODEL_NAME=READER_MODEL_NAME):
    """Returns a cached instance of the Open Source LLM."""
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)#, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=100,
    )

    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

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
def answer_with_rag(question: str, llm: Pipeline, knowledge_index: FAISS, prompt: str, num_retrieved_docs: int = 2) -> Tuple[str, List[LangchainDocument]]:
    try:
        # Gather documents with retriever
        with st.spinner("reading Antoine's resumé..."):
            print("=> Retrieving documents...")
            relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
            relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

            # Build the final prompt
            context = "\nExtracted documents:\n"
            context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

            final_prompt = prompt.format(question=question, context=context)

        # Redact an answer
        with st.spinner("processing your query..."):
            print("=> Generating answer...")
            answer = llm(final_prompt)[0]["generated_text"]
            print("------------------------------------------------------------------------")
            print(answer)
            print("------------------------------------------------------------------------")
            #add_message(stream_response(answer))
            add_message(answer)

    except Exception as e:
        st.error(f"Error during query processing: {e}")
        print(f"Debug: {e}")

# Main application logic
def main():
    """Main function to run the application logic."""
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []
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