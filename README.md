# Baseline RAG EvalLLM

![Baseline RAG Architecture EvalLLM](docs/RAG.png)

This repository was forked from another RAG architecture: https://github.com/yourusername/UltimateAdvisor.git 

## 🚀 **Key Features**

- **📚 Intelligent Document Processing**: Automatically indexes and processes docs provided.
- **🤖 AI-Powered Q&A**: Ask natural language questions and get accurate, context-aware answers
- **📊 Source Attribution**: Every answer includes relevant source documents with similarity scores and page references

## 🛠 **Technology Stack**

### **Backend (Python)**
- **FastAPI**: High-performance API framework with automatic OpenAPI documentation
- **SQLModel**: Modern Python SQL toolkit combining SQLAlchemy + Pydantic
- **LlamaIndex**: RAG framework for document processing and querying
- **PostgreSQL + pgvector**: Vector database for embeddings storage
- **Ollama**: Local LLM serving (supports Llama 3.2, Mistral, etc.)

### **Frontend (TypeScript/React)**
- **React 19**: Modern React with latest features
- **Vite**: Lightning-fast build tool
- **TailwindCSS**: Utility-first CSS framework
- **SWR**: Data fetching with caching and revalidation
- **Radix UI**: Accessible, unstyled UI components

### **Infrastructure**
- **Apptainer**: Multi-container orchestration
- **pgvector**: PostgreSQL extension for vector operations
- **uv**: Fast Python package management

### Data Flow

1. **Document Processing**: PDF documents are chunked and embedded using Ollama
2. **Vector Storage**: Embeddings are stored in PostgreSQL with pgvector extension
3. **Query Processing**: User questions are embedded and matched against stored vectors
4. **Response Generation**: Retrieved context is sent to the chat model for answer generation
5. **History Tracking**: All conversations are persisted for future reference
