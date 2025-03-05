import os
import streamlit as st
import shutil
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from populate_database import fetch_content, split_document, add_to_chroma, clear_database, CHROMA_PATH
from embedding_model import get_embedding_function
from dotenv import load_dotenv

# Set page configuration FIRST - before any other Streamlit commands
st.set_page_config(page_title="News Researcher", layout="wide")

# Load environment variables
load_dotenv()

# Check for database deletion marker - without using streamlit commands directly
def check_deletion_marker():
    messages = []
    
    marker_file = os.path.join(os.path.dirname(CHROMA_PATH), "delete_chroma.marker")
    
    if os.path.exists(marker_file):
        # Wait a moment to ensure file locks are released
        import time
        time.sleep(2)
        
        try:
            # Database isn't open yet, so deletion should work
            if os.path.exists(CHROMA_PATH):
                # Try to handle locked files
                try:
                    shutil.rmtree(CHROMA_PATH)
                except PermissionError:
                    # If it fails due to permission, create a notice for manual deletion
                    messages.append(("error", "Database files are still locked. Please manually delete the 'chroma' folder and restart."))
                    # We'll still try to remove the marker so we don't get stuck in a loop
            
            # Remove the marker file regardless of whether deletion worked
            if os.path.exists(marker_file):
                os.remove(marker_file)
                
            # Only report success if we actually deleted the folder
            if not os.path.exists(CHROMA_PATH):
                messages.append(("success", "Database cleared successfully on startup"))
            return True, messages
        except Exception as e:
            if os.path.exists(marker_file):
                try:
                    os.remove(marker_file)
                except:
                    pass
            messages.append(("error", f"Failed to clear database on startup: {str(e)}"))
    return False, messages

# Call the function
db_cleared, messages = check_deletion_marker()

# Initialize session state to store processed URLs
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []

# Page header
st.title("📰 News Researcher")
st.markdown("Enter news article URLs and ask questions about them.")

# Display any messages from the database check
for msg_type, msg_text in messages:
    if msg_type == "success":
        st.success(msg_text)
    elif msg_type == "error":
        st.error(msg_text)
    else:
        st.info(msg_text)

# URL input section
with st.expander("News Sources", expanded=True):
    with st.form("url_form"):
        url1 = st.text_input("URL 1", placeholder="https://example.com/news/article1")
        url2 = st.text_input("URL 2", placeholder="https://example.com/news/article2")
        url3 = st.text_input("URL 3", placeholder="https://example.com/news/article3")
        
        process_col, clear_col = st.columns([3, 1])
        with process_col:
            process_button = st.form_submit_button("Process Articles")
        with clear_col:
            clear_button = st.form_submit_button("Clear Database")
    
    if process_button:
        urls = [url for url in [url1, url2, url3] if url]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            with st.spinner("Fetching and processing articles..."):
                # Fetch content from URLs
                documents = fetch_content(urls)
                if documents:
                    # Split into chunks
                    chunks = split_document(documents)
                    # Store in vector DB
                    add_to_chroma(chunks)
                    st.session_state.processed_urls = urls
                    st.success(f"Processed {len(documents)} articles successfully!")
                else:
                    st.error("No content could be retrieved from the provided URLs.")
    
    if clear_button:
        with st.spinner("Clearing database..."):
            clear_database()
            st.session_state.processed_urls = []

# Question answering section
st.markdown("---")
st.subheader("Ask Questions")

if st.session_state.processed_urls:
    st.success(f"Ready to answer questions on {len(st.session_state.processed_urls)} articles")
    
    question = st.text_input("What would you like to know about these articles?")
    
    if question:
        if os.path.exists(CHROMA_PATH):
            with st.spinner("Researching an answer..."):
                # Set up the retrieval system
                embeddings = get_embedding_function()
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 5})
                
                # Set up the QA model
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile"
                )
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
                
                # Run the query
                try:
                    response = qa_chain.invoke({"query": question})
                    
                    # Display the answer
                    st.markdown("### Answer:")
                    st.write(response["result"])
                    
                    # Display sources
                    with st.expander("Sources"):
                        for doc in retriever.invoke(question):
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"**Excerpt:** {doc.page_content[:300]}...")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("No data in the database. Please process some URLs first.")
else:
    st.info("Please enter and process some news article URLs to get started.")

# Footer
st.markdown("---")
st.caption("News Researcher v0.1 - Powered by LangChain and Groq")