import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from policy_loader import load_policy_documents
from rag import build_rag_chain


def main():
    documents = load_policy_documents()
    print(f"Loaded {len(documents)} policy documents.")

    chain = build_rag_chain(documents)

    print("\nUniversity IT Policy Assistant (qwen3-0.6b)")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()