"""
Vector store implementation for product search.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from data.mock_data import MOCK_PRODUCTS

# Set tokenizers parallelism to false to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

class ProductVectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.products = []
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        self.index_file = 'data/product_index.faiss'
        self.products_file = 'data/products.pkl'
        
        # Initialize Groq
        self.llm = ChatGroq(
            temperature=0.7,
            model="llama3-8b-8192"
        )
        
        # Create prompt template for product summaries
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful shopping assistant. Summarize the search results in a natural way, highlighting key features and relevance to the query."),
            ("user", "{context}")
        ])
        
        self.load_or_create_index()

    def load_or_create_index(self):
        """Initialize or load existing FAISS index"""
        if os.path.exists(self.index_file) and os.path.exists(self.products_file):
            # Load existing index and products
            self.index = faiss.read_index(self.index_file)
            with open(self.products_file, 'rb') as f:
                self.products = pickle.load(f)
        else:
            # Create new index
            embedding_size = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(embedding_size)
            # Add mock products for testing
            self._add_mock_products()

    def _add_mock_products(self):
        """Add mock products for testing"""
        for product in MOCK_PRODUCTS:
            self.add_product(product)

    def add_product(self, product_info: Dict):
        """Add a product to the vector store"""
        # Create a text representation of the product
        text = f"{product_info['name']} {product_info['description']} Category: {product_info['category']}"
        
        # Get embedding
        embedding = self.model.encode([text])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Store product info
        self.products.append(product_info)
        
        # Save to disk
        self._save_to_disk()

    def search(self, query: str, k: int = 3) -> Dict:
        """
        Search for similar products and generate AI summary
        Returns dict with results and AI-generated summary
        """
        # Get query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in FAISS
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Get matched products
        results = []
        for idx in I[0]:
            if idx < len(self.products):
                results.append(self.products[idx])

        # Generate AI summary if results found
        if results:
            summary = self._generate_summary(query, results)
        else:
            summary = "No products found matching your query."

        return {
            "results": results,
            "summary": summary
        }

    def _generate_summary(self, query: str, results: List[Dict]) -> str:
        """Generate an AI summary of the search results using Groq"""
        # Create context for LLM
        context = f"Query: {query}\n\nFound products:\n"
        for product in results:
            context += f"- {product['name']}: {product['description']} (${product['price']})\n"

        try:
            # Use LangChain with Groq
            chain = self.summary_prompt | self.llm
            response = chain.invoke({"context": context})
            return response.content
        except Exception as e:
            # Fallback to basic summary if AI generation fails
            return f"Found {len(results)} products matching your query. Top result: {results[0]['name']}"

    def _save_to_disk(self):
        """Save the index and products to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.products_file, 'wb') as f:
            pickle.dump(self.products, f)

# Create a global instance
vector_store = ProductVectorStore() 