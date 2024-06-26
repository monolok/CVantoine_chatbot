import os
import streamlit as st
import time
import numpy as np
import json
#from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from faiss import IndexFlatL2, read_index

@st.cache_resource
def get_client():
    """Returns a cached instance of the MistralClient."""
    #load_dotenv()
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)

CLIENT: MistralClient = get_client()

# PROMPT = """
# An excerpt from a document is given below.

# ---------------------
# {context}
# ---------------------

# Given the document excerpt, answer the following query.
# If the context does not provide enough information, decline to answer.
# Do not output anything that can't be answered from the context.

# Query: {query}
# Answer:
# """

PROMPT = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query in the same language as the query.
Do not output anything that can't be answered from the context.
If the context does not provide enough information suggest the following:
- Schedule a video call with Antoine Bertin for more detailed information. You can do this by visiting his Calendly link: 'https://calendly.com/antoinebertin/30'.
- Learn more about Antoine's story, you can download the 'antoine' package. To do this, use the command 'pip install antoine' in your terminal and run the 'hire_antoine' command.
Query: {query}
Answer:
"""

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

# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    try:
        for r in response:
            content = r.choices[0].delta.content
            # prevent $ from rendering as LaTeX
            content = content.replace("$", "\$")
            yield content
    except Exception as e:
        yield f"Error: {e}"

# Decorator to cache the embedding computation
@st.cache_data
def embed(text: str):
    """Returns the embedding for a given text, caching the result."""
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding

# Function to build and cache the index from PDFs in a directory
@st.cache_resource
def load_and_cache_index():
    """Loads and caches the faiss index and chunks from JSON file."""
    index = read_index("data/vector_cv.index")
    with open("data/cv_chunks.json", "r", encoding="utf-8") as f:
        chunk_dict = json.load(f)
    return index, chunk_dict

# Function to reply to queries using the built index
def reply(query: str, index: IndexFlatL2, chunks):
    """Generates a reply to the user's query based on the indexed PDF content."""
    try:
        embedding = embed(query)
        embedding = np.array([embedding])

        _, indexes = index.search(embedding, k=2)
        context = [chunks[f"{i}"] for i in indexes.tolist()[0]]

        messages = [
            ChatMessage(role="user", content=PROMPT.format(context=context, query=query))
        ]
        response = CLIENT.chat_stream(model="mistral-medium", messages=messages)

        # Wait for a short time to ensure the response is ready
        # Display a loading indicator while processing
        with st.spinner("processing your query..."):
            time.sleep(2)
        add_message(stream_response(response))

    except Exception as e:
        st.error(f"Error during query processing: {e}")
        print(f"Debug: {e}")

# Main application logic
def main():
    """Main function to run the application logic."""
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []
    
    try:
        index, chunks = load_and_cache_index()
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
        reply(query, index, chunks)

if __name__ == "__main__":
    main()