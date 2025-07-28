import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Paths and setup
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o")

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
               "which might reference context in the chat history, "
               "formulate a standalone question which can be understood "
               "without the chat history. Do NOT answer the question, just "
               "reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use "
               "the following pieces of retrieved context to answer the "
               "question. If you don't know the answer, just say that you "
               "don't know. Use three sentences maximum and keep the answer "
               "concise.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit UI
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📚 RAG Chat with LangChain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat box
user_input = st.chat_input("Ask a question")

if user_input:
    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(SystemMessage(content=result["answer"]))

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
