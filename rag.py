from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


def build_rag_chain(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/models/sentence-transformers--all-MiniLM-L6-v2/snapshots/master",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatOllama(model="qwen3:0.6b", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are a university IT support assistant.
Answer the question using ONLY the context provided below.
If the context does not contain the answer, you MUST respond with:
"I don't know. The provided policy documents do not contain that information."
Do NOT invent phone numbers, emails, or any other details.
Do NOT provide any unrelated instructions.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain