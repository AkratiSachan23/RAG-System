# Retrieval-Augmented Generation (RAG) Q&A for National-Security Focused

This notebook implements a Retrieval-Augmented Generation (RAG) system for Question Answering on a collection of PDF documents, with a focus on national security topics. The goal is to provide a system that can answer questions based on the information contained within the provided documents.
RAG Explanation Diagram: https://excalidraw.com/#json=rlEuyKtv59B1LpvuwNSkp,W8K5Idk3QjLXGui5840lDw

## Process and Thinking Behind the Solution

The solution follows a standard RAG pipeline, which involves several key steps:

1.  **Install Dependencies**: Necessary libraries like `faiss-cpu`, `sentence-transformers`, `transformers`, and `PyMuPDF` are installed to handle vector indexing, text embedding, language modeling, and PDF processing respectively.

2.  **Load your PDFs from `./data/`**: The notebook is designed to read PDF files placed in a `./data/` folder. The `load_pdfs` function uses `PyMuPDF` (fitz) to extract text content from each page of the PDFs. The text from all pages of a single document is concatenated.

3.  **Extract and clean text**: While explicit text cleaning beyond basic extraction is not extensively shown, the `load_pdfs` function handles the initial extraction of text from the PDF format. Further cleaning (like removing headers, footers, or special characters) could be added if necessary based on the specific document types.

4.  **Chunk into passages**: Large text documents are split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter` from `langchain`. This is crucial because language models have input token limits, and smaller chunks allow for more relevant context to be retrieved. Overlapping chunks help to preserve continuity of information across chunk boundaries.

5.  **Build a FAISS index**: The text chunks are converted into numerical representations called embeddings using a pre-trained sentence transformer model (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text. A FAISS (Facebook AI Similarity Search) index is then built on these embeddings. FAISS is a library for efficient similarity search and clustering of dense vectors, enabling fast retrieval of relevant chunks based on a query's embedding.

6.  **Run an interactive Q&A loop with FLAN-T5**:
    *   **Retrieval**: When a user asks a question, the question is also embedded using the same sentence transformer model. This query embedding is then used to search the FAISS index for the `TOP_K` most similar chunk embeddings. The corresponding text chunks are retrieved as context.
    *   **Generation**: The retrieved chunks and the user's question are provided as input to a pre-trained language model (`google/flan-t5-base`). The language model then generates an answer based on the provided context. The prompt is structured to encourage the model to use the document snippets and cite its sources.

## Setup and Usage

1.  **Clone the repository or download the notebook.**
2.  **Create a folder named `data` in the same directory as the notebook.**
3.  **Place your PDF files within the `data` folder.**
4.  **Run the notebook in Google Colab or a compatible Python environment.**
5.  **Ensure all dependencies are installed by running the first code cell.**
6.  **Execute the subsequent cells in order.**
7.  **Once the Q&A loop starts, type your questions at the `Q>` prompt and press Enter.**
8.  **Type `exit` or `quit` to end the Q&A session.**

This notebook provides a basic yet effective RAG system for querying your own documents. You can experiment with different embedding models, language models, chunk sizes, and retrieval parameters to optimize performance for your specific use case.
