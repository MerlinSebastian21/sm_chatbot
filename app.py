
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 1. PAGE CONFIGURATION & STYLES ---
st.set_page_config(
    page_title="Simelabs AI Chatbot",
    page_icon="💡",
    layout="centered"
)

# --- 2. SIDEBAR CONTENT ---
with st.sidebar:
    st.image("assets/simelabs_logo.jpg", width=200)
    st.title("💡 Simelabs AI Chatbot")
    st.markdown("This chatbot is your intelligent assistant, powered by documents from Simelabs. It can answer questions about services, careers, and company information.")
    
    st.session_state.show_sources = st.toggle("Show Sources", value=True, help="Display the source documents used to generate the answer.")

    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Simelabs AI Chatbot. How may I assist you?"}]
        # st.session_state.history_store = {}
        st.rerun()

# --- 3. LOAD ENVIRONMENT VARIABLES & SECRETS ---
try:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
except TypeError: # This will trigger if .env is not found
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("API keys or index name are missing! Please set them in your .env file or Streamlit secrets.")
    st.stop()

# --- 4. CACHED BACKEND SETUP ---
@st.cache_resource
def get_backend_components():
    """Initializes and returns the RAG components (LLM, Embeddings, Retriever)."""
    print("--- Initializing chatbot components (should run only once) ---")
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    
    # Use MMR for diverse and comprehensive results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 8, 'fetch_k': 20}
    )
    print("--- Vector store and retriever are ready. ---")
    return llm, retriever

# --- 5. CREATE THE STATEFUL RAG CHAIN ---
@st.cache_resource
def get_rag_chain(_llm, _retriever):
    """Creates and returns the stateful RAG chain."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(_llm, _retriever, contextualize_q_prompt)

    system_prompt = (
        "You are the 'Simelabs Chatbot', an expert assistant for question-answering. "
        "Use ONLY the following pieces of retrieved context to answer the question. Your tone should be professional, helpful, and confident. "
        "If the context does not contain the answer, state that based on the provided documents, you couldn't find the information. Do not use any outside knowledge. "
        "Be conversational and explain what you did find, if relevant."
        "\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 6. SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Simelabs Chatbot. How may I assist you?"}]

if "history_store" not in st.session_state:
    st.session_state.history_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

# --- 7. CHAT INTERFACE LOGIC ---
st.header("Simelab's Chatbot", divider="rainbow")

# Get cached components
llm, retriever = get_backend_components()
rag_chain = get_rag_chain(llm, retriever)

# Wrap RAG chain with memory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the message
        if "sources" in message and message["sources"] and st.session_state.show_sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"Source {i+1}:\n\n{source.page_content}")
                    if source.metadata:
                        st.caption(f"Metadata: {source.metadata}")


# Display suggested questions if chat is new
if len(st.session_state.messages) <= 1:
    st.markdown("---")
    st.markdown("**Suggested Questions:**")
    q_col1, q_col2 = st.columns(2)
    
    questions = [
        "What are the core services of Simelabs?",
        "Tell me about recent job openings.",
        "What are Simelabs' healthcare solutions?"
    ]

    for i, q in enumerate(questions):
        # Distribute questions into columns
        col = q_col1 if i % 2 == 0 else q_col2
        if col.button(q, use_container_width=True):
            st.session_state.prompt_from_button = q
            st.rerun()

# Handle user input
prompt = st.chat_input("Ask a question...")

# Use the prompt from a button click if it exists
if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
    prompt = st.session_state.prompt_from_button
    st.session_state.prompt_from_button = None # Reset after use

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("..."):
            config = {"configurable": {"session_id": "streamlit_session"}}
            response = conversational_rag_chain.invoke({"input": prompt}, config=config)
            answer = response["answer"]
            sources = response["context"]

        # Simulate typing effect
        for chunk in answer.split():
            full_response += chunk + " "
            time.sleep(0.04)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Add the full response and sources to the session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "sources": sources
    })
