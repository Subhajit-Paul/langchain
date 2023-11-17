import streamlit as st
from PyPDF2 import PdfReader
import pickle, os
from langchain.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
inference_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    length_function=len)


r = {
    "GPT 2": "gpt2",
    "Falcon 7B": "tiiuae/falcon-7b",
    "RedPajama Chat 3B": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    "Google Flan T5 XXL": "google/flan-t5-xxl",
    "Google Flan T5 Large": "google/flan-t5-large",
    "Google MT-5 Large": "google/mt5-large",
    "Google UMT5 Small": "google/umt5-small",
    "Google Seahorse": "google/seahorse-large-q6"
    }
with st.sidebar:
    st.title("Welcome!")
    val = st.radio(
        "Choose a Language Model",
        ("GPT 2",
         "Falcon 7B",
         "RedPajama Chat 3B",
         "Google Flan T5 XXL",
         "Google Flan T5 Large",
         "Google MT-5 Large",
         "Google UMT5 Small",
         "Google Seahorse",
         )
    )
    repo_id = r[val]

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name=repo_id)
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 256})
chain = load_qa_chain(llm=llm, chain_type="stuff")

pdf = st.file_uploader(label="Choose PDF to Upload", type="pdf")
if pdf is not None:
    r = PdfReader(pdf)
    text = ""
    for page in r.pages:
        text += page.extract_text()
    chunks = text_splitter.split_text(text)
    
    if os.path.exists(f"{pdf.name[:-4]}.pkl"):
        with open(f"{pdf.name[:-4]}.pkl", "rb") as f:
            store = pickle.load(f)
    else:
        store = faiss.FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{pdf.name[:-4]}.pkl", "wb") as f:
            pickle.dump(store, f)
    query = st.text_input("Ask Questions: ")
    if query:
        docs = store.similarity_search(query=query, k=1)
        
        resp = chain.run(input_documents=docs, question=query)
        st.write(resp)
    