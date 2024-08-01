import os
import time
import pickle
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Web Article QA Tool 📈")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vectorStoreDB.pkl"

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(
    model='gemini-pro', google_api_key=os.getenv("GEMINI_API_KEY"))

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...Please Wait")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...just a minute")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {"batch_size": 32}
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    vectorStoreDB = FAISS.from_documents(docs, instructor_embeddings)
    main_placeholder.text(
        "Embedding Vector Started Building...Please Wait for a while")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorStoreDB, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorStoreDB = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorStoreDB.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                # Split the sources by newline
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
