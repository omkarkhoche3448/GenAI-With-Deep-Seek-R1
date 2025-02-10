import streamlit as st
import time
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color:rgba(206, 173, 28, 0.91) !important;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "📌 Choose AI Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    
    st.markdown("### GEN-AI Capabilities")
    st.markdown("""
    - 📄 Advanced Document Analysis  
    - 🔎 Context-Based Information Retrieval  
    - 🧠 Intelligent Query Responses  
    """)
    
    st.divider()
    
    st.markdown("💡 Powered by [Ollama](https://ollama.ai/) & [LangChain](https://python.langchain.com/)")


PDF_STORAGE_PATH = 'document_store/pdfs/'
TYPING_SPEED = 0.01  
PROMPT_TEMPLATE = """
You are an elite research analyst and document specialist with deep expertise in information extraction, analysis, and explanation. Your role is to provide precise, insightful, and well-structured responses based on the provided document context.

CONTEXT ANALYSIS:
{document_context}

USER QUERY:
{user_query}

RESPONSE GUIDELINES:
1. First, analyze the context thoroughly and identify key information relevant to the query
2. Structure your response in a clear, logical manner
3. Use professional, academic language while maintaining clarity
4. Support your statements with specific references from the context
5. Highlight any numerical data, statistics, or key findings
6. If any part of the query cannot be answered from the context, explicitly state this

RESPONSE FORMAT:
- Primary Answer: Provide a direct, comprehensive answer to the main query (2-3 sentences)
- Key Details: List any relevant supporting information from the context
- Confidence Level: Indicate your confidence in the response (High/Medium/Low) based on context relevance
- Limitations: Note any important caveats or missing information

QUALITY CHECKS:
- Ensure factual accuracy based solely on the provided context
- Verify that all statements are supported by the document
- Confirm that the response directly addresses the user's query
- Check that technical terms are used accurately and appropriately

IMPORTANT NOTES:
- If the context doesn't contain relevant information, state: "Based on the provided context, I cannot provide a reliable answer to this specific query."
- For partial information, clarify what aspects you can and cannot answer
- Maintain objectivity and avoid speculation beyond the provided context
- If technical terms are used, provide brief clarifications where necessary

Please format your response as follows:

📝 Main Answer:
[Your primary response here]

🔍 Supporting Details:
[Key supporting information]

⚖️ Confidence: [High/Medium/Low]

ℹ️ Additional Notes:
[Any important caveats, limitations, or clarifications]

Begin your analysis now. Remember to be precise, factual, and informative while maintaining clarity and professionalism.
"""

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


def typewriter_effect(response_text):
    response_placeholder = st.empty()
    displayed_text = ""
    
    for char in response_text:
        displayed_text += char
        response_placeholder.markdown(f"**🤖 DocuMind AI:** {displayed_text}")
        time.sleep(TYPING_SPEED)  


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant 🤖")
st.markdown("---")

uploaded_pdf = st.file_uploader(
    "📂 Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    st.info("📄 Processing your document...")
    
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("💬 Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("🤖 Thinking..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        with st.chat_message("assistant", avatar="🤖"):
            typewriter_effect(ai_response)
