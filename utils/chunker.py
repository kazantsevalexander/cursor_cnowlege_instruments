"""
Text chunking utilities for processing large documents.
"""

from typing import List, Optional
import tiktoken
from loguru import logger


class TextChunker:
    """
    Utility class for splitting text into chunks suitable for embedding.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            logger.debug(f"Text fits in single chunk ({total_tokens} tokens)")
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < total_tokens:
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start index, accounting for overlap
            if end_idx == total_tokens:
                break
            start_idx = end_idx - self.chunk_overlap
        
        logger.info(f"Split text into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks
    
    def chunk_documents(self, documents: List[str]) -> List[dict]:
        """
        Chunk multiple documents and track their source.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of dictionaries with 'text', 'doc_id', and 'chunk_id'
        """
        chunked_docs = []
        
        for doc_id, doc_text in enumerate(documents):
            chunks = self.chunk_text(doc_text)
            
            for chunk_id, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks)
                })
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} total chunks")
        return chunked_docs
    
    @staticmethod
    def simple_split(
        text: str,
        max_length: int = 1000,
        separator: str = "\n\n"
    ) -> List[str]:
        """
        Simple text splitting by separator and max length.
        
        Args:
            text: Text to split
            max_length: Maximum character length per chunk
            separator: String to split on
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by separator first
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= max_length:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

