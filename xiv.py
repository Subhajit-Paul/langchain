import streamlit as st
import arxiv
import pickle, os, requests, io
from PyPDF2 import PdfReader
from langchain.vectorstores import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
inference_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, length_function=len)


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

link = st.text_input(label="Paste Arxiv DOI here", placeholder="https://doi.org/10.48550/arXiv.1706.03762")
if link != "":
    with st.spinner('Paper Data being fetched'):
        id = "".join(link.split('.')[3]+"."+link.split('.')[4])
        client = arxiv.Client()
        search = arxiv.Search(id_list=[id])
        paper = next(arxiv.Client().results(search))
        pdf_link = paper.pdf_url+".pdf"
    st.title(paper.title, anchor=False)
    authorsd = ""
    for i in paper._raw.authors:
        authorsd += i.name + ", "
    st.subheader(authorsd[:-2], anchor=False)
    st.subheader("", divider='rainbow', anchor=False)
    st.subheader("Summary", anchor=False)
    st.caption(paper.summary)
    st.subheader("", divider='rainbow', anchor=False)
    with st.spinner("Paper is being Downloaded"):
        resp = requests.get(pdf_link)
        open_pdf_file = io.BytesIO(resp.content)
        reader = PdfReader(open_pdf_file)
    st.success('Paper Downloaded!')
    with st.spinner("Knowledge Base is being created"):
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        chunks = text_splitter.create_documents(text)   
        if os.path.exists(f"{paper.title}.pkl"):
            with open(f"{paper.title}.pkl", "rb") as f:
                store = pickle.load(f)
        else:
            text_embeddings = embeddings.embed_documents(chunks)
            text_embedding_pairs = zip(chunks, text_embeddings)
            text_embedding_pairs_list = list(text_embedding_pairs)
            store = faiss.FAISS.from_embeddings(text_embedding_pairs_list, embeddings)
            with open(f"{paper.title}.pkl", "wb") as f:
                pickle.dump(store, f)
    st.success('Knowledge Base Created')
    query = st.text_input("Ask Questions: ")
    if query:
        docs = store.similarity_search(query)
        with st.spinner("Answer being Computed"):
            resp = chain.run(input_documents=docs, question=query)
        st.balloons()
        st.write(resp)
    