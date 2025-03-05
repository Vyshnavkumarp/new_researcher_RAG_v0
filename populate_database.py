import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from embedding_model import get_embedding_function

CHROMA_PATH = "chroma"

def fetch_content(urls):
    """Fetch content from URLs using requests and BeautifulSoup"""
    content = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in urls:
        try:
            st.info(f"Fetching content from {url}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav']):
                script_or_style.decompose()
                
            # Extract article content
            article = soup.find('article')
            if article:
                main_content = article
            else:
                # If no article tag, try to find main content area
                main = soup.find('main')
                if main:
                    main_content = main
                else:
                    # Fallback to body content
                    main_content = soup.body
            
            # Get text and convert to markdown-like format
            text = ""
            
            # Extract headings with hierarchy
            for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'blockquote']):
                if tag.name.startswith('h'):
                    level = int(tag.name[1])
                    text += '#' * level + ' ' + tag.get_text(strip=True) + '\n\n'
                elif tag.name == 'p':
                    text += tag.get_text(strip=True) + '\n\n'
                elif tag.name == 'blockquote':
                    text += '> ' + tag.get_text(strip=True) + '\n\n'
                elif tag.name in ['ul', 'ol']:
                    for li in tag.find_all('li'):
                        text += '* ' + li.get_text(strip=True) + '\n'
                    text += '\n'
            
            if text:
                content.append(Document(page_content=text, metadata={"source": url}))
                st.success(f"Successfully processed {url}")
            else:
                st.warning(f"No content extracted from {url}")
                
        except Exception as e:
            st.error(f"Failed to fetch {url}: {str(e)}")
    
    return content

def split_document(documents: list[Document]):
    # initialize the text splitter (RecursiveCharacterTextSplitter) and split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # initialization of vector database (Chroma)
    db = Chroma(
        persist_directory = CHROMA_PATH, embedding_function = get_embedding_function()
    )
    db.add_documents(chunks)
    return db

def clear_database():
    """Mark database for deletion on next application startup"""
    import os
    
    # Create a marker file instead of trying to delete while running
    marker_file = os.path.join(os.path.dirname(CHROMA_PATH), "delete_chroma.marker")
    
    try:
        # Create a marker file
        with open(marker_file, 'w') as f:
            f.write("delete")
        
        st.success("Database marked for deletion")
        st.warning("To complete database clearing, please:")
        st.info("1. Close this browser tab\n2. Stop the Streamlit server with Ctrl+C in your terminal\n3. Run 'streamlit run app.py' again")
            
    except Exception as e:
        st.error(f"Error marking database for deletion: {str(e)}")