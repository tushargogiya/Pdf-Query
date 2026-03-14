import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are an expert document analyst. Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't have enough information in this document to answer that."
Never make up information. Cite the relevant page when possible.

Context:
{context}

Question: {question}

Detailed Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# Singleton embeddings model (loaded once, reused across sessions)
@staticmethod
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class PdfQAEngine:
    _embeddings = None  # class-level cache

    def __init__(self):
        self.vectordb = None
        self.qa_chain = None
        self.doc_info = {}

        if PdfQAEngine._embeddings is None:
            PdfQAEngine._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )

    def load_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        """Load PDF bytes, chunk, embed, build FAISS index."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        self.vectordb = FAISS.from_documents(chunks, PdfQAEngine._embeddings)

        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15},
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        self.doc_info = {
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks),
        }

        os.unlink(tmp_path)
        return self.doc_info

    def answer(self, question: str) -> dict:
        if not self.qa_chain:
            raise ValueError("No PDF loaded. Please upload and process a PDF first.")

        result = self.qa_chain({"query": question})

        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page", 0)
            snippet = doc.page_content[:250].strip()
            key = (page, snippet[:50])
            if key not in seen:
                seen.add(key)
                sources.append({"page": int(page) + 1, "snippet": snippet})

        return {
            "answer": result["result"],
            "sources": sources,
        }
