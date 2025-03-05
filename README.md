# News Researcher

## Overview
News Researcher is a powerful tool that allows users to extract meaningful insights from multiple news articles using AI. Simply provide URLs to news articles, and the application will process the content, store it in a vector database, and answer your questions based on the information in those articles.

![News Researcher](https://i.imgur.com/placeholder-image.png)

## Features

- **Web Scraping**: Extract content from any news website
- **Vector Database**: Store article content for efficient retrieval
- **AI-Powered Q&A**: Ask questions and get answers based on the articles
- **Source Attribution**: See which parts of articles were used to generate answers
- **Simple Interface**: Easy-to-use Streamlit interface for all operations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news_researcher_v0.git
   cd news_researcher_v0
   ```

2. Create a virtual environment:
   ```bash
   python -m venv news_venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     news_venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source news_venv/bin/activate
     ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup

This application requires two API keys:

1. **Google API Key**: For embedding generation
   - Get a key from [Google Cloud Platform](https://console.cloud.google.com/)
   - Enable the Generative AI API service

2. **Groq API Key**: For the language model
   - Sign up at [Groq](https://console.groq.com/) to get your API key

Create a .env file in the root directory with:
```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter up to three news article URLs in the provided input fields

4. Click "Process Articles" to extract and store the content

5. Once processing is complete, type your question in the "Ask Questions" section

6. View the AI-generated answer and check the sources used to create it

7. To clear the database and start fresh, click "Clear Database" and restart the application

## How It Works

### Web Scraping
The application uses BeautifulSoup to fetch and parse content from provided URLs, focusing on article content and removing unnecessary elements like navigation bars and advertisements.

### Text Processing
Content is split into manageable chunks using LangChain's RecursiveCharacterTextSplitter to ensure optimal processing.

### Vector Embeddings
Google's Generative AI embedding model converts text chunks into vector representations that capture the semantic meaning of the content.

### Vector Storage
Embeddings are stored in a Chroma vector database for efficient similarity searching.

### Question Answering
When you ask a question:
1. Your question is converted to an embedding
2. The most relevant text chunks are retrieved from the database
3. Groq's LLM (Llama 3.3 70B Versatile) generates an answer based on those chunks
4. Sources are displayed so you know which parts of articles were used

## Technical Architecture

The application consists of these main components:

- app.py: Main Streamlit interface and application logic
- populate_database.py: Functions for web scraping and database management
- embedding_model.py: Configuration for the Google embedding model
- chroma: Directory where the vector database is stored

## Limitations

- Web scraping might not work perfectly on all news sites due to varying HTML structures
- Some websites may block automated scraping
- Processing very large articles or many articles at once may take time
- Answers are generated based only on the content provided; the AI doesn't have knowledge beyond what's in the articles

## Troubleshooting

- **Database deletion issues**: On Windows, if you encounter file locking issues when clearing the database, follow the instructions to restart the application completely
- **API key errors**: Ensure your API keys are correctly added to the .env file and are valid
- **Web scraping failures**: Some websites use techniques to prevent scraping; try different news sources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Vector embeddings powered by [Google Generative AI](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
- LLM capabilities provided by [Groq](https://groq.com/)
- Framework orchestration by [LangChain](https://www.langchain.com/)

---

© 2025 News Researcher | [GitHub](https://github.com/yourusername/news_researcher_v0)