import os
import fitz  # PyMuPDF - Used for reading PDF files
import numpy as np
import faiss  # Facebook AI Similarity Search - For fast vector indexing and search
import pickle  # For saving and loading Python objects (our documents and sources)
import json
from sentence_transformers import SentenceTransformer  # For generating text embeddings
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
import requests  # For making HTTP requests to the Ollama API


class RAGPipeline:
    """
    Implements a complete Retrieval-Augmented Generation (RAG) pipeline.
    This class handles:
    1. Loading documents (PDFs, TXTs) from a folder.
    2. Splitting them into manageable chunks.
    3. Generating numerical vector embeddings for each chunk.
    4. Storing these embeddings in a FAISS vector index for fast search.
    5. Providing a query interface that retrieves relevant chunks,
       builds a prompt, and uses an LLM (via Ollama) to generate an answer.
    6. Handling dynamic addition of new files.
    """

    def __init__(self, data_folder="data/sample_docs", index_folder="data/faiss_index"):
        """
        Initializes the RAG pipeline.
        Sets up file paths, loads the embedding model, and either loads an
        existing index or builds one if documents are present.
        """
        self.data_folder = data_folder
        self.index_folder = index_folder
        # Ensure the data and index directories exist
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.index_folder, exist_ok=True)

        # Load the sentence transformer model. This will download if not present.
        # 'all-MiniLM-L6-v2' is a good, fast model for embedding generation.
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Lists to store the text chunks and their corresponding source (filename, chunk_index)
        self.documents: List[str] = []
        self.doc_sources: List[Tuple[str, int]] = []

        # The FAISS index object. Will be None until loaded or built.
        self.index = None

        # --- Smart Initialization Logic ---
        # 1. Try to load an existing index from disk
        if not self._load_index():
            # 2. If loading fails, check if there are documents to index
            if os.listdir(self.data_folder):
                print("No index found, building from existing documents...")
                # 3. Load all documents, build the index, and save it
                self._load_documents()
                self._build_index()
                self._save_index()
            else:
                # 4. If no index and no docs, wait for a file upload
                print("No documents found in data folder. Index will be created upon first upload.")

    def _load_documents(self):
        """
        Loads all documents from the `data_folder`, extracts text,
        splits into chunks, and populates `self.documents` and `self.doc_sources`.
        """
        self.documents.clear()
        self.doc_sources.clear()

        print(f"Loading documents from {self.data_folder}...")
        for filename in os.listdir(self.data_folder):
            # Skip hidden files (like .DS_Store)
            if filename.startswith("."):
                continue

            filepath = os.path.join(self.data_folder, filename)
            text = ""

            # Handle .txt files
            if filename.endswith(".txt"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    print(f"[TXT] Loaded {filename}, {len(text)} characters")
                except Exception as e:
                    print(f"Error reading TXT {filename}: {e}")

            # Handle .pdf files
            elif filename.endswith(".pdf"):
                text = self._read_pdf(filepath)
                print(f"[PDF] Loaded {filename}, {len(text)} characters")

            # If text extraction failed or file was empty, skip it
            if not text.strip():
                continue

            # Split the extracted text into chunks
            chunks = self._split_into_chunks(text)

            # Store each chunk and its source
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                # Store a tuple: (filename, chunk_number)
                self.doc_sources.append((filename, i))

        print(f"✅ Total chunks indexed: {len(self.documents)}")

    def _read_pdf(self, path):
        """Helper function to extract text from a PDF file using PyMuPDF (fitz)."""
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return ""

    def _split_into_chunks(self, text, chunk_size=500, chunk_overlap=50):
        """
        Splits a long text into smaller, overlapping chunks.
        This is crucial for RAG because:
        1. Embeddings are more meaningful on smaller, focused text segments.
        2. It allows us to fit relevant context into the LLM's limited context window.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Max size of each chunk
            chunk_overlap=chunk_overlap,  # Overlap between chunks to maintain context
            length_function=len,  # Function to measure chunk size
            separators=["\n\n", "\n", " ", ""],  # How to split the text
        )
        return text_splitter.split_text(text)

    def _build_index(self):
        """
        Builds the FAISS vector index from the loaded documents.
        1. Encodes all text chunks into vectors (embeddings).
        2. Creates a FAISS index.
        3. Adds the vectors to the index.
        """
        if not self.documents:
            raise ValueError("No documents to index.")

        print("Building FAISS index...")
        # 1. Encode all documents into vectors. This can be time-consuming.
        embeddings = self.embedding_model.encode(self.documents).astype("float32")

        # 2. Get the dimension of the embeddings (e.g., 384 for 'all-MiniLM-L6-v2')
        dim = embeddings.shape[1]

        # 3. Create a FAISS index. IndexFlatL2 is a simple index that performs
        #    an exhaustive search (calculates L2 distance to all vectors).
        #    It's fast and effective for millions of vectors.
        self.index = faiss.IndexFlatL2(dim)

        # 4. Add the document embeddings to the index
        self.index.add(embeddings)
        print("📦 FAISS index built successfully")

    def _save_index(self):
        """Saves the FAISS index and the document/source lists to disk."""
        if self.index:
            # Save the FAISS index
            faiss.write_index(
                self.index, os.path.join(self.index_folder, "faiss.index")
            )

            # Save the documents list
            with open(os.path.join(self.index_folder, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)

            # Save the sources list
            with open(os.path.join(self.index_folder, "sources.pkl"), "wb") as f:
                pickle.dump(self.doc_sources, f)
            print("💾 FAISS index and documents saved.")

    def _load_index(self):
        """
        Loads the FAISS index, documents, and sources from disk.
        All three files must exist to successfully load.
        """
        try:
            index_path = os.path.join(self.index_folder, "faiss.index")
            docs_path = os.path.join(self.index_folder, "documents.pkl")
            sources_path = os.path.join(self.index_folder, "sources.pkl")

            # Check if all required files exist
            if not all(os.path.exists(p) for p in [index_path, docs_path, sources_path]):
                print("❌ One or more index files are missing.")
                return False

            # Load the FAISS index
            self.index = faiss.read_index(index_path)

            # Load the documents (the text chunks)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

            # Load the sources (the metadata for each chunk)
            with open(sources_path, "rb") as f:
                self.doc_sources = pickle.load(f)

            print("✅ FAISS index and documents loaded from disk.")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False

    def _process_and_add_document(self, filename: str):
        """
        Processes a *single* new document, generates embeddings,
        and adds them to the *existing* FAISS index.
        This is for incremental updates.
        """
        filepath = os.path.join(self.data_folder, filename)
        text = ""

        # 1. Read the new file
        if filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"[TXT] Processing new file {filename}, {len(text)} characters")
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")

        elif filename.endswith(".pdf"):
            text = self._read_pdf(filepath)
            print(f"[PDF] Processing new file {filename}, {len(text)} characters")

        if not text.strip():
            print(f"❌ Skipped empty file: {filename}")
            return

        # 2. Split into chunks
        chunks = self._split_into_chunks(text)

        # 3. Generate embeddings for the new chunks
        new_embeddings = self.embedding_model.encode(chunks).astype("float32")

        # 4. Add the new embeddings to the existing index
        self.index.add(new_embeddings)

        # 5. Append the new chunks and sources to the in-memory lists
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.doc_sources.append((filename, i))

        print(f"➕ Added {len(chunks)} new chunks from {filename}.")

    def _retrieve(self, query, top_k=2):
        """
        Retrieves the top_k most relevant document chunks for a given query.
        1. Encodes the query into a vector.
        2. Searches the FAISS index for the k-nearest neighbors.
        3. Returns the corresponding text chunks and their sources.
        """
        if self.index is None:
            raise ValueError("Index is not loaded or built.")

        # 1. Encode the user's query into a vector
        query_emb = self.embedding_model.encode([query]).astype("float32")

        # 2. Search the index. Returns distances (D) and indices (I)
        # We only need the indices (I)
        _, indices = self.index.search(query_emb, top_k)

        # 3. Map the retrieved indices back to the original documents and sources
        return [(self.documents[i], self.doc_sources[i]) for i in indices[0]]

    def query(self, question):
        """
        Main user-facing query method.
        Orchestrates the full RAG flow: Retrieve -> Augment -> Generate.
        """
        # 1. Retrieve: Find relevant documents
        retrieved = self._retrieve(question)

        # 2. Augment: Combine the retrieved chunks into a single context string
        context = "\n".join([doc for doc, _ in retrieved])

        # 3. Augment: Create the prompt, instructing the LLM to use the context
        prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {question}
Answer:"""

        # 4. Generate: Send the prompt to the LLM and get an answer
        answer = self._generate(prompt)

        # Return the answer and the sources it was based on
        return {
            "answer": answer,
            "sources": [{"filename": f, "chunk": c} for _, (f, c) in retrieved],
        }

    def _generate(self, prompt):
        """
        Sends the augmented prompt to an LLM (Ollama) and returns the response.
        """
        # This is the non-streaming version
        try:
            # Make a POST request to the Ollama server's /api/generate endpoint
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:1b", "prompt": prompt, "stream": False},
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx, 5xx)

            data = response.json()
            return data.get("response", "").strip()  # Extract the 'response' field

        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            print(f"❌ Error during Ollama API call: {e}")
            return f"Error: Failed to connect to Ollama server. Please ensure the model is running. {e}"

    def add_file_and_reindex(self, filename: str, content: bytes):
        """
        Public method to add a new file to the RAG system.
        1. Saves the file to the data_folder.
        2. Handles two cases:
           a) No index exists: Builds a new index from all files.
           b) Index exists: Adds the new file's chunks to the existing index.
        3. Saves the updated index.
        """
        path = os.path.join(self.data_folder, filename)

        # 1. Save the new file to the data folder
        try:
            with open(path, "wb") as f:
                f.write(content)
            print(f"📁 Uploaded and saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to write file {filename}: {e}")
            return {"status": "error", "message": str(e)}

        # 2. Check if this is the first file (no index)
        if self.index is None:
            print("No existing index. Building from scratch...")
            # This is the first file, so build the *entire* index
            self._load_documents()
            self._build_index()
        else:
            # 3. If index exists, just add the new document (incremental update)
            try:
                self._process_and_add_document(filename)
            except Exception as e:
                print(f"❌ Failed to process and index new file {filename}: {e}")
                # If processing fails, remove the bad file
                os.remove(path)
                return {"status": "error", "message": f"Failed to process file: {e}"}

        # 4. Save the updated index to disk
        self._save_index()
        return {"status": "success", "message": f"{filename} uploaded and indexed."}