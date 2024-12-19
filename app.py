from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import base64
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import fitz
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
def get_response_model(question):
    model=AzureChatOpenAI(
    azure_endpoint=azure_endpoint,  openai_api_version=version,
    deployment_name=deployment_name,
        openai_api_type="azure",
        temperature=0.1,
        max_tokens=4096,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        top_p=0.01)
    response=model.invoke(question)
    return response
# Background CSS
st.set_page_config(page_title="PDF Talker", page_icon="🧊",layout="wide",initial_sidebar_state="expanded",)
def load_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
import streamlit as st

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)

st.header("Your personal pdf reader")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    def extract_text_from_pdf(pdf_file):
        pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in pdf_reader:
            text += page.get_text()
        pdf_reader.close()
        return text

    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("### Extracted PDF Content:")
    st.text_area("PDF Text", pdf_text, height=300)
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
    all_splits = text_splitter.split_text(pdf_text)
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",azure_endpoint=azure_endpoint)
    document_search = FAISS.from_texts(all_splits, embeddings)
    input=st.text_input("Ask questions from pdf",key="input")
    submit=st.button("Ask the question")
    if submit:
        query = input
        docs = document_search.similarity_search(query)
        print(docs)
        if not docs:
            raise ValueError("No documents found for the given query.")
        input_data = {"input_documents": docs, "question": query}
        chain = load_qa_chain(AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                openai_api_version=version,
                deployment_name=deployment_name,
                openai_api_key=openai_api_key,
                openai_api_type="azure",
                temperature=0.1,
                max_tokens=4096,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                top_p=0.01,
            ), chain_type="stuff")
        response = chain.run(**input_data)
        print(response)
        st.subheader("The response is")
        st.write(response)
