import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 InsuraChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

    # User input text box
    user_input = st.text_input("Your message:", key="user_input")

def main():
    st.header("Chat with InsuraChat💬")

    # Read PDF and extract text
    pdf_path = "dataset.pdf"
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Load or generate embeddings
    store_name = pdf_path[:-4]
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Accept user query
    query = st.session_state.user_input

    # Perform question answering
    if query:
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct") 
        docs = VectorStore.similarity_search(query=query, k=3)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
        
        # Append user query and AI response to session state
        st.session_state.messages.append(HumanMessage(content=query))
        st.session_state.messages.append(AIMessage(content=response))

    # Display message history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.image("female.png", width=40)
            st.text_area("You:", value=msg.content, key=msg.content)
        elif isinstance(msg, AIMessage):
            st.image("robot.png", width=40)
            st.text_area("InsuraChat:", value=msg.content, key=msg.content)

if __name__ == '__main__':
    main()
