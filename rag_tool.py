## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents

documents = load_policy_documents()


def rag_tool(question: str) -> str:
    """
    Query the university IT policy knowledge base.
    Note: this knowledge base does NOT contain any phone number.
    """
    from rag import build_rag_chain
    chain = build_rag_chain(documents)
    return chain.invoke(question)