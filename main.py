#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import os
import re
import nest_asyncio
import uvicorn
import traceback
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate 
from prometheus_client import Counter, CollectorRegistry, multiprocess



class QueryRequest(BaseModel):
    query: str

# Initialize FastAPI
app = FastAPI()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load website data (Brainlox courses)
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

# Preprocess Text
def clean_text(doc):
    text = doc.page_content
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^A-Za-z0-9\s.,()-]', '', text)
    text = text.lower()
    doc.page_content = text
    return doc

cleaned_documents = [clean_text(doc) for doc in documents]

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(cleaned_documents)

# Embeddings & FAISS vector storage
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Chat history
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    output_key="answer",
    return_messages=True
)

# Course registration links and details
course_details = {
    "Artificial Intelligence (for Kids)": {
        "description": "An introductory AI course for children to learn fundamental AI concepts in a fun way.",
        "duration": "4 weeks",
        "certification": "Yes",
        "link": "https://brainlox.com/courses/intro-ai-kids"
    },
    "AI Essentials Bootcamp": {
        "description": "Designed for beginners and professionals to understand core AI concepts, including ML, neural networks, and real-world applications.",
        "duration": "6 weeks (live + recorded sessions)",
        "certification": "Yes",
        "link": "https://brainlox.com/courses/ai-bootcamp"
    },
    "AI in Stock Market": {
        "description": "A specialized course covering AI applications in trading, predictive analytics, and financial forecasting.",
        "duration": "5 weeks",
        "certification": "Yes",
        "link": "https://brainlox.com/courses/ai-stock-market"
    }
}

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a fact-based AI assistant representing Brainlox. Your responses must:
    - Use a numbered list format.
    - Be strictly fact-based, concise, and context-aware.
    - Remain consistent in structure and tone.
    - Use "We" instead of "I."

    **Context:** {context}  
    **User Question:** {question}  

    **Response (Strictly Numbered Format):**
    """
)

topic_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Enhanced SQL Injection Detection
def is_sql_injection(query):
    sql_injection_patterns = [
        r"(\bselect\b|\bunion\b|\binsert\b|\bupdate\b|\bdrop\b|\bdelete\b|\b--\b|\b;--\b|\b'or\b|\b' or \b'1'='1\b|\bselect\s.*\sfrom\s.*)",
        r"\b--\b",  # Handle comments
        r"\bselect\b",  # Detect SELECT statements
        r"\bunion\b",  # Detect UNION statements
        r"\binsert\b",  # Detect INSERT statements
        r"\bupdate\b",  # Detect UPDATE statements
        r"\bdrop\b",  # Detect DROP statements
        r"\bdelete\b",  # Detect DELETE statements
        r"\b' or \b1=1",  # Catch ' OR 1=1 injections
        r"';--",  # Handle statements with single quote and comments
        r"or 1=1",  # OR 1=1 injections
        r"select.*from.*",  # Generic SELECT FROM pattern
        r"(\b'|\b\")\s*or\s*1=1\s*--",  # Detect ' OR 1=1 --
        r"(\b'|\b\")\s*or\s*\w*\=\w*\s*--",  # Detect ' OR a=a --
        r"(\b'|\b\")\s*or\s*'a'='a'\s*--",  # Detect 'a'='a' injections
        r"(\b'|\b\")\s*or\s*1=1#.*",  # Handle OR 1=1 with comment
        r"(\b'|\b\")\s*or\s*'x'='x'\s*--",  # Handle 'x'='x' injections
        r"\bunion\b.*\bselect\b",  # Handle UNION SELECT queries
        r"select\s.*\sfrom\s.*--",  # Handle select from queries
        r"drop\s*table\s*\w+\s*--",  # Handle DROP TABLE injections
        r"\b1=1;\s*--",  # Catch cases like 1=1; --
        r"(\b'|\b\")\s*and\s*\d*=\d*;--",  # Detect AND conditions with SQL injections
        r"';\s*--",  # SQL injections ending with a semicolon and comment
        r"^.*\bselect\b.*\bwhere\b.*--",  # Select with WHERE condition
        r"^.*\bselect\b.*\bfrom\b.*\bwhere\b.*--"  # Complex Select From Where injections
    ]

    return any(re.search(pattern, query.lower()) for pattern in sql_injection_patterns)

# Enhanced XSS Detection
def is_xss_injection(query):
    xss_patterns = [
        r"<script.*?>.*?</script>",  # Match <script> tags
        r"javascript:.*",  # Match javascript: pseudo-protocol
        r"on\w+\s*=\s*['\"]?\s*.*alert.*",  # Match event handler injections (e.g., onerror, onclick)
        r"<img\s+[^>]*onerror\s*=\s*['\"]?\s*alert.*"  # Match <img> tag with onerror
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in xss_patterns)

# Command Injection Prevention
def is_command_injection(query):
    dangerous_patterns = [r"[;&|><]", r"\b(rm\s+-rf|wget|curl|nc|netcat|bash|sh)\b"]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in dangerous_patterns)

# Detect booking intent
def detect_booking_intent(query):
    booking_patterns = [
        r"\bregister\b", r"\bsign[ -]?up\b", r"\benroll\b", r"\bbooking\b", 
        r"\bjoin\b", r"\bapply\b", r"\bsubscription\b", r"\bhow to (register|enroll|sign up|book)\b",
        r"\bget started\b", r"\bstart learning\b", r"\bwant to learn\b", r"\bneed to sign up\b",r"\bregistration\b"

    ]
    return any(re.search(pattern, query.lower()) for pattern in booking_patterns)

# Detect acknowledgment messages
def detect_acknowledgment(query):
    acknowledgments = ["thank you", "thanks", "ok", "got it", "appreciate it"]
    return any(word in query.lower() for word in acknowledgments)

# Generate Response
def generate_response(query):
    query_lower = query.lower().strip()
        
    if is_sql_injection(query):
        return "1. Query blocked due to security concerns (SQL Injection detected)."
    
    if is_xss_injection(query):
        return "1. Query blocked due to security concerns (XSS Injection detected)."
    
    if is_command_injection(query):
        return "1. Query blocked due to security concerns (Command Injection detected)."
    
    if detect_booking_intent(query):
        return "1. Kindly visit Website for Course Enrollment/Booking Demo/Course Registration."
    
    # Handle empty query
    if not query_lower:
        return "1. Please provide a valid question."
    
    # General acknowledgment handling
    if detect_acknowledgment(query):
        return "1. You're welcome! Let us know if you have any other questions."

    # Greetings Handling
    if query_lower in ["hello", "hi", "hey"]:
        return "1. Hello! How can we assist you today?"

    if query_lower in ["how are you", "how are you?", "how are you doing?"]:
        return "1. We're here to assist you. How can we help?"

    # Check if user is asking about a course
    for course, details in course_details.items():
        if course.lower() in query_lower:
            return f"""
            1. {details['description']}
            2. Duration: {details['duration']}
            3. Certification: {details['certification']}
            4. Register here: {details['link']}
            """
    
    # Retrieve response from LLM
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    raw_response = topic_chain.run({"question": query, "chat_history": chat_history})
    
    return refine_response(raw_response)

# Format Response
def refine_response(response_text):
    lines = response_text.strip().split("\n")
    numbered_list = [f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip()]
    return "\n".join(numbered_list)

# API Endpoints
@app.get("/")
def home():
    return {"message": "Chatbot API is Running!"}

@app.post("/query")
def query(request: QueryRequest):
    try:
        final_response = generate_response(request.query)
        return {"response": final_response.split("\n")}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
# Configure Logging
logging.basicConfig(level=logging.ERROR, filename="error.log", filemode="a", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Exception Handler for Validation Errors (Malformed JSON, Missing Fields)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request format. Please check your input JSON."}
    )

# Generic Exception Handler for Internal Errors
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Internal Server Error: {traceback.format_exc()}")  # Logs the full error traceback
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please try again later."}
    )


# Allow FastAPI to Run Inside Jupyter/Colab
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# In[ ]:




