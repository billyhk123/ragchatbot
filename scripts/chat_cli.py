from app.chain import build_chain

def main():
    chain = build_chain()
    print("RAG chatbot (Poe). Type 'exit' to quit.")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        a = chain.invoke(q)
        print("\nBot:", a)

if __name__ == "__main__":
    main()
