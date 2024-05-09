from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import streamlit as st
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

vclient = voyageai.Client(api_key="pa-5If_Qj0bm3R5LUP3pdvW5os4BroeR4gLfW1HN6tx-7Q")
QDRANT_API_KEY = "JYOvctbqxEA64YyHdHkKcHGNVG5_5b_W-ngeCubQzUbLvNKD1Fnwow"
port = 6333
qclient = QdrantClient(
    url = "https://51083629-1391-4b2d-aaf4-f6084ad812c1.us-east4-0.gcp.cloud.qdrant.io",
    port = port,
    api_key = QDRANT_API_KEY,
)
with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="chatbot_api_key", type="password")

st.title("💬 Chatbot")
st.caption("🚀 RAG powered chat")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def getMessageType(message, index):
    if message["role"] == "user":
        return HumanMessage(message["prompt_content"])
    else:
        if index == 0:
            return SystemMessage(message["content"])
        return AIMessage(message["content"])

# Embed the query
def embed_query(query: str):
    embed_response = vclient.embed(
        query,
        model = "voyage-large-2",
        input_type = "query",
        )
    return embed_response.embeddings[0]

def qdrant_search(bedding : list):
    knowledge = qclient.search(
        collection_name = "test_collection",
        query_vector = bedding,
    )
    return knowledge

# Prompt modified to include context
def custom_prompt(query: str):
  knowledge = qdrant_search(embed_query(query))
  augment_prompt = f"""Using the contexts below, answer the query:

  Contexts:
  {knowledge}

  Query: {query}"""
  return augment_prompt  

if prompt := st.chat_input():
    if not anthropic_api_key:
        st.info("Please add your Anthropic API key to continue.")
        st.stop()

    client = ChatAnthropic(api_key=anthropic_api_key, model_name="claude-3-opus-20240229")
    st.session_state.messages.append({"role": "user", "content": prompt, "prompt_content": custom_prompt(prompt)})
    st.chat_message("user").write(prompt)
    response = client.invoke([getMessageType(i,ind) for ind, i in enumerate(st.session_state.messages)])
    msg = response.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
