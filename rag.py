import cohere
import time
import streamlit as st

class RetrieverWithRerank:
    def __init__(self, retriever, api_key):
        self.retriever = retriever
        self.co = cohere.Client(api_key=api_key)

    def retrieve(self, query: str, top_n: int):
        # First call to the retriever fetches the closest indices
        nodes = self.retriever.retrieve(query)
        nodes = [
            {
                "text": node.node.text,
                "llamaindex_id": node.node.id_,
            }
            for node
            in nodes
        ]
        # Call co.rerank to improve the relevance of retrieved documents
        reranked = self.co.rerank(query=query, documents=nodes, model="rerank-english-v3.0", top_n=top_n)
        nodes = [nodes[node.index] for node in reranked.results]
        return nodes

def format_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"System: {message['content']}\n"
        elif message["role"] == "user":
            prompt += f"User: {message['content']}\n"
    return prompt

# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)

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