# importing necessary packages
from langchain_community import document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import re
import numpy as np


class DataIngestion:
    def __init__(self, pdf_path):
        
        project_root = Path(__file__).resolve().parents[1]
        self.pdf_path = project_root / pdf_path
        self.data = None
        self.model = SentenceTransformer("BAAI/bge-m3")

    def data_loading(self):

        # Langchain method of loading data
        loader = document_loaders.PyPDFLoader(self.pdf_path)
        self.data = loader.load()
        return self.data

    def data_cleaning(self):

        # Data Loading
        self.data_loading()
        # Cleaning the documents
        for i in self.data:
            text = i.page_content
            text = re.sub(r"\n+", "\n", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"^Page\s*\d+\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
            text = text.strip()
            i.page_content = text
        return self.data
    
    def data_chunking(self, characters = 1000, overlap=150):

        # Data Cleaning
        self.data_cleaning()

        splitter = RecursiveCharacterTextSplitter(chunk_size=characters, chunk_overlap=overlap)
        self.chunks = splitter.split_documents(self.data)
        return self.chunks

    def data_embedding(self):
        
        # Data Chunking
        self.data_chunking()
        
        text = [chunk.page_content for chunk in self.chunks]
        self.embeddings = self.model.encode(text, batch_size= 16)
        self.embeddings = np.array(self.embeddings, dtype = "float32")
        faiss.normalize_L2(self.embeddings)

        self.metadata = []
        for idx, chunk in enumerate(self.chunks):
            self.metadata.append({
            "id": f"chunk_{idx}",
            "text": chunk.page_content,
            "metadata": chunk.metadata
            })

        return self.embeddings, self.metadata