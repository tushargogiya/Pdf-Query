import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

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


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class PdfQAEngine:
    _embeddings = None  # class-level singleton to avoid reloading

    def __init__(self):
        self.vectordb = None
        self.retriever = None
        self.rag_chain = None
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
        """Load PDF bytes, chunk, embed, build FAISS index and LCEL chain."""
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
        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15},
        )

        # Modern LCEL chain — retrieves docs + generates answer
        self.rag_chain = RunnableParallel(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: _format_docs(x["context"])
                )
                | PROMPT
                | self.llm
                | StrOutputParser()
            ),
            context=lambda x: x["context"],
        )

        self.doc_info = {
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks),
        }

        os.unlink(tmp_path)
        return self.doc_info

    def answer(self, question: str) -> dict:
        if not self.rag_chain:
            raise ValueError("No PDF loaded. Please upload and process a PDF first.")

        retrieved_docs = self.retriever.invoke(question)
        result = self.rag_chain.invoke({"context": retrieved_docs, "question": question})

        sources = []
        seen = set()
        for doc in retrieved_docs:
            page = doc.metadata.get("page", 0)
            snippet = doc.page_content[:250].strip()
            key = (page, snippet[:50])
            if key not in seen:
                seen.add(key)
                sources.append({"page": int(page) + 1, "snippet": snippet})

        return {
            "answer": result["answer"],
            "sources": sources,
        }
