from phase1 import build_knowledge_base, answer_question, summarize_pdf_hierarchical

if __name__ == "__main__":

    pdf_path = "document.pdf"  # Make sure file exists

    # Extract + preprocess ONCE
    text = extract_text_from_pdf(pdf_path)
    sentences = nltk.sent_tokenize(text)
    sentences = clean_sentences(sentences)

    print("Building knowledge base...")
    chunks, embeddings = build_knowledge_base(sentences)

    print("PDF loaded successfully!\n")

    print("\n📌 Document Summary:\n") #shows summary

    summary = summarize_pdf_hierarchical(sentences)

    for s in summary:
        print("-", s)

    print("\nYou can now ask questions about the document.\n")

    #question loop
    while True:

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        top_chunks = answer_question(question, chunks, embeddings, top_k=2)

        if not top_chunks:
            print("No relevant content found in the document.")
            continue

        print("\n📖 Relevant content from document:\n")

        for chunk in top_chunks:
            print("-", chunk)
            print()

