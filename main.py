import os
import time

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def main():
    # Loader file from environment variable
    loader = UnstructuredPDFLoader(os.getenv("PDF_FILE_PATH", "book.pdf"))
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(pages, embeddings).as_retriever()
    while True:
        query = input("Please enter your question (type 'exit' to quit):").strip()
        if query == "exit":
            break

        print('Processing, please wait...')
        start_time = time.time()
        docs = docsearch.get_relevant_documents(query)
        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        output = chain.run(input_documents=docs, question=query)
        elapsed_time = time.time() - start_time

        print(output)
        print(f"Processing complete, duration：{elapsed_time:.2f}s\n")


if __name__ == '__main__':
    main()
