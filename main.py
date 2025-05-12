#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#---------------------------- System & Utility Modules -------------------------------------------------------
import os
import re
import pickle
import logging
import traceback
import datetime
import boto3
import zipfile

#---------------------------  Async and Server Runtime --------------------------------------------------------
import nest_asyncio
import uvicorn

#--------------------------- Email Handling ------------------------------------------------------------------
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

#-------------------------- Date Parsing --------------------------------------------------------------------
import dateparser

#-------------------------  Environment Variables ------------------------------------------------------------
from dotenv import load_dotenv

#-------------------------  Validation & Typing ------------------------------------------------------------
from pydantic import BaseModel
from typing import Optional

#------------------------- FastAPI Framework -----------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

#------------------------- Monitoring (Prometheus) ------------------------------------------------------------
from prometheus_client import Counter, CollectorRegistry, multiprocess

#------------------------- Google Calendar API ---------------------------------------------------------------
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

#------------------------- Language Detection ----------------------------------------------------------------
from langdetect import detect_langs, detect

#------------------------ Hugging Face Transformers ----------------------------------------------------------
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
from transformers import T5Tokenizer

#------------------------ LangChain Core --------------------------------------------------------------------
from langchain.prompts import PromptTemplate 
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

#------------------------ LangChain: Document Handling & Embeddings ------------------------------------------
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

#--------------------- Define Input Schema for User Query using Pydantic ----------------------------------
class QueryRequest(BaseModel):
    query: str

#----------------------- Initialize FastAPI ----------------------------------------------------------------
app = FastAPI(debug=True)

#---------------------- Load environment variables ---------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SCOPES = ['https://www.googleapis.com/auth/calendar']

#--------------------- Global conversation state -----------------------------------------------------------
conversation_state = {
    "awaiting_email": False,
    "awaiting_time": False,
    "email": "",
}

#--------------------- Load website data (Brainlox courses) -------------------------------------------------
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

#---------------------- Preprocess Text ----------------------------------------------------------------------
def clean_text(doc):
    text = doc.page_content
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^A-Za-z0-9\s.,()-]', '', text)
    text = text.lower()
    doc.page_content = text
    return doc

cleaned_documents = [clean_text(doc) for doc in documents]

#--------------------- Split text into chunks -----------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(cleaned_documents)

#---------------------- Embeddings & FAISS vector storage -----------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()

#-----------------------Initialize LLM -------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

#------------------------- Chat history --------------------------------------------------------------------
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    output_key="answer",
    return_messages=True
)

#--------------------------- Course registration links and details ----------------------------------------------------
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

#--------------------------------- Custom prompt -------------------------------------------------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a fact-based AI assistant representing Brainlox. Please follow these guidelines strictly:
1. Respond only with verified facts — no opinions or assumptions.
2. Use a **numbered list** format (1., 2., 3., etc.).
3. Be **concise**, **relevant**, and **aware of the provided context**.
4. Maintain a consistent tone of professionalism.
5. Refer to the organization using **"We"** instead of "I".
6. Do not provide empty responses — if no data is available, state that clearly.
7. Always include a dollar symbol ($) before any fees or prices.

📘 **Context:**  
{context}

❓ **User Question:**  
{question}

🧾 **Response (Start with 1.):**
"""
)

#----------------- Initialize Conversational Retrieval Chain with Custom Prompt for Context-Aware Responses--------------
topic_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# --- Constants ---
BUCKET_NAME = "mohanesh-chatbot-models"
MODEL_KEY = "sarcasm_model_v3.zip"
LOCAL_ZIP_PATH = "sarcasm_model_v3.zip"
LOCAL_MODEL_DIR = "sarcasm_model_v3"

# --- Function: Download from S3 and Extract ---
def download_and_extract_model(bucket_name, s3_key, local_zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        print(f"Downloading {s3_key} from S3 bucket {bucket_name}...")
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_key, local_zip_path)

        print(f"Extracting model to {extract_dir}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Model extraction complete.")
    else:
        print("Model already exists locally.")

# --- Run Download ---
download_and_extract_model(BUCKET_NAME, MODEL_KEY, LOCAL_ZIP_PATH, LOCAL_MODEL_DIR)

# --- Load Model ---
print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
model.eval()
print("Model loaded.")

# --- Prediction Function ---
def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction == 1  # True if sarcastic

'''
# S3 Bucket info
BUCKET_NAME = "mohanesh-chatbot-models"
MODEL_KEY = "sarcasm_model_v3.zip"
LOCAL_MODEL_PATH = "sarcasm_model_v3.zip"
LOCAL_MODEL_DIR = "sarcasm_model_v3"

# Function to download model from S3
def download_model_from_s3(bucket_name, model_key, local_path):
    if not os.path.exists(local_path):
        print("Downloading model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, model_key, local_path)
        
        # Unzip if it's a zip file
        if local_path.endswith('.zip'):
            print("Extracting model files...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(local_path))
            print("Extraction complete.")

# Check if model exists locally, otherwise download
if not os.path.exists(LOCAL_MODEL_DIR):
    download_model_from_s3(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)

# Load the trained model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)

# Set model to evaluation mode
model.eval()

# Define sarcasm detection function
def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction == 1  # True for sarcastic, False for not sarcastic'''

#-------------------- Define rude phrases detection function -------------------------------------------------
def rude_sentiment_detected(text):
    text = text.lower()
    rude_stems = [
        "wast",     # waste, wasting, wasted
        "useless",  
        "terribl",  # terrible, terribly
        "worst",
        "pointless",
        "disaster",
        "regret",
        "frustrat", # frustrated, frustrating
        "unhelpful",
        "stupid",
        "meaningless",
        "garbage",
        "nonsense"
    ]
    return any(stem in text for stem in rude_stems)



#--------------------------- offense detection function --------------------------------------------------------------
def is_offensive_query(query):
    """
    Checks if the input query contains offensive language.
    Returns True if offensive, False otherwise.
    """
    offensive_keywords = [
        "stupid", "idiot", "dumb", "hate", "fool", "shut up", "nonsense",
        "kill", "ugly", "bastard", "damn", "hell", "screw", "moron"
    ]

    query_lower = query.lower()
    return any(word in query_lower for word in offensive_keywords)

#----------------------------- List of vague expressions ---------------------------------------------------------------
VAGUE_QUERIES = [
    r"\bwhat\s*now\b", r"\bokay\b", r"\band then\b", r"\bhelp\b", r"\bhello\b",
    r"\bhey\b", r"\bhmm+\b", r"\byes\b", r"\bno\b", r"\byup\b", r"\bsure\b",
    r"\btell me\b", r"\bnext\b", r"\bcontinue\b"
]

#--------------------------- Vague detection function -----------------------------------------------------------------
def is_vague_query(query):
    return any(re.search(pattern, query.lower()) for pattern in VAGUE_QUERIES)

#--------------------------- gibberish detection function --------------------------------------------------------------
def is_gibberish(query):
    return bool(re.fullmatch(r"[a-zA-Z]{5,}", query)) and len(set(query)) < 15

#--------------------------- Enhanced SQL Injection Detection ---------------------------------------------------------
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

#----------------------------- Enhanced XSS Detection -----------------------------------------------------------
def is_xss_injection(query):
    xss_patterns = [
        r"<script.*?>.*?</script>",  # Match <script> tags
        r"javascript:.*",  # Match javascript: pseudo-protocol
        r"on\w+\s*=\s*['\"]?\s*.*alert.*",  # Match event handler injections (e.g., onerror, onclick)
        r"<img\s+[^>]*onerror\s*=\s*['\"]?\s*alert.*"  # Match <img> tag with onerror
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in xss_patterns)

#------------------------------- Command Injection Prevention -------------------------------------------------------
def is_command_injection(query):
    dangerous_patterns = [r"[;&|><]", r"\b(rm\s+-rf|wget|curl|nc|netcat|bash|sh)\b"]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in dangerous_patterns)

#-------------------------------- Detect booking intent ----------------------------------------------------------------
def detect_booking_intent(query):
    booking_patterns = [
        r"\bregister\b", r"\bsign[ -]?up\b", r"\benroll\b", r"\bbooking\b", 
        r"\bjoin\b", r"\bapply\b", r"\bsubscription\b", r"\bhow to (register|enroll|sign up|book)\b",
        r"\bget started\b", r"\bstart learning\b", r"\bwant to learn\b", r"\bneed to sign up\b",r"\bregistration\b", r"\bbook (a )?demo\b",
        r"\bschedule (a )?demo\b"

    ]
    return any(re.search(pattern, query.lower()) for pattern in booking_patterns)

#---------------------- Google Calendar API Authentication and Service Initialization ---------------------------------
def get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret_853349819074-5afkt8v8ce9k17setl98t11klg18pi0s.apps.googleusercontent.com.json", SCOPES)
            creds = flow.run_console()
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

calendar_service = get_calendar_service()

#------------------ Google Meet Invite Creation with Calendar Event Scheduling ------------------------------------
def create_google_meet_invite(guest_email, start_dt):
    end_dt = start_dt + datetime.timedelta(minutes=30)

    event = {
        'summary': 'Demo Session Booking',
        'description': 'Demo session with AI Sales Assistant.',
        'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Kolkata'},
        'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Kolkata'},
        'attendees': [{'email': guest_email}],
        'conferenceData': {
            'createRequest': {
                'requestId': f"meet-{datetime.datetime.utcnow().timestamp()}",
                'conferenceSolutionKey': {'type': 'hangoutsMeet'},
            }
        }
    }

    event = calendar_service.events().insert(
        calendarId='primary',
        body=event,
        conferenceDataVersion=1,
        sendUpdates='all'
    ).execute()

    return event.get('hangoutLink', 'No Meet link generated.')

#------------------------- Send Confirmation Email with Google Meet Link to Attendee -----------------------------------
def send_meeting_email(to_email, meet_link):
    sender_email = "bargemohanesh@gmail.com"
    app_password = os.getenv("APP_PASSWORD")

    subject = "Your Demo Session is Scheduled!"
    body = f"""
    Hi there,

    Your demo has been successfully booked. 🎉
    Here’s your Google Meet link:

    👉 {meet_link}

    See you at the scheduled time!

    Regards,  
    AI Sales Assistant 🤖
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, to_email, msg.as_string())
            print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

        
#-------------------------------- Detect acknowledgment messages ----------------------------------------------------
def detect_acknowledgment(query):
    acknowledgments = ["thank you", "thanks", "ok", "got it", "appreciate it"]
    return any(word in query.lower() for word in acknowledgments)

#---------------------------------- Format Response -----------------------------------------------------------------
def refine_response(response_text):
    return response_text.strip().split("\n")


#----------------------------------- Generate Response ---------------------------------------------------------------
def generate_response(query): 
    query_lower = query.lower().strip()

    #---------------------------- Handle empty query ------------------------
    if not query_lower:
        return "You Provided empty query, please provide a valid question."

    #---------------------------- Greetings handling --------------------------
    if query_lower in ["hello", "hi", "hey"]:
        return "Hi there! How can I assist you today?"
    
    #---------------------------- Generic Que. handling --------------------------
    if query_lower in ["how are you", "how are you?", "how are you doing?"]:
        return "I'm functioning at full capacity! How can I help?"

     #---------------------------- Sarcasm handling ---------------------------------
    if detect_sarcasm(query) or rude_sentiment_detected(query):
        return "We're sorry you feel that way. We'd love to know how we can improve!"
    
    #-------------------------- General acknowledgment handling ----------------
    if detect_acknowledgment(query):
        return "You're welcome! Let us know if you have any other questions."
   
    #------------------------- Offensive language handling --------------------------
    if is_offensive_query(query):
        return "let's be respectful, kindly rephrase you query"

    #------------------------- SQL Injection attacks handling ----------------------------
    if is_sql_injection(query):
        return "Query blocked due to security concerns (SQL Injection detected)."
    
    #------------------------- XSS Injection attacks handling ----------------------------
    if is_xss_injection(query):
        return "Query blocked due to security concerns (XSS Injection detected)."
    #------------------------ Command Injection attacks handling ----------------------------
    if is_command_injection(query):
        return "Query blocked due to security concerns (Command Injection detected)."

    #-------------------------- Gibberish queries handling -----------------------------------
    if is_gibberish(query):
        return "Could you please rephrase your question?"
    #--------------------------- vague queries handling -------------------------------------
    if is_vague_query(query):
        return "Can you please specify your interest? For example: Python, AI, Java, HTML or Cloud Computing."
    #--------------------------- legitimacy queries handling -------------------------------------
    if "is brainlox legitimate" in query_lower or "brainlox scam" in query_lower or "brainlox fake" in query_lower or "brainlox genuine" in query_lower:
        return (
        "1. Brainlox offers online courses in AI, coding, and cloud computing.\n"
        "2. The platform provides detailed course structures, pricing, and contact support.\n"
        "3. Legal pages like Terms & Privacy Policy are available on the website.\n"
        "4. We're a verified online learning platform. "
        "However, for full trust, we recommend checking reviews or contacting us directly."
    )

    #---------------------------- Course  -------------------------------------------------------
    for course, details in course_details.items():
        if course.lower() in query_lower:
            return f"""
            1. {details['description']}
            2. Duration: {details['duration']}
            3. Certification: {details['certification']}
            4. Register here: {details['link']}
            """

    #--------------------------- Fallback to LLM ------------------------------------------------
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    raw_response = topic_chain.run({"question": query, "chat_history": chat_history})
    
    return refine_response(raw_response)

#------------------------------------ API Endpoints ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "Chatbot API is Running!"}

#--------------------------------- Main Chatbot Endpoint to Handle Booking Flow and General Queries --------------------------
@app.post("/query")
def query(request: QueryRequest):
    try:
        user_input = request.query.strip()

        #---------------- Step 1: Booking intent -----------------------------------------------------------
        if detect_booking_intent(user_input):
            conversation_state.update({"awaiting_email": True, "awaiting_time": False, "email": ""})
            return {"response": ["🗓️ Sure! Please share your Gmail ID to book your demo session."]}

        #----------------- Step 2: Email collection --------------------------------------------------------
        elif conversation_state["awaiting_email"]:
            if re.match(r"[^@]+@gmail\.com", user_input):  # stricter: only Gmail
                conversation_state["email"] = user_input
                conversation_state.update({"awaiting_email": False, "awaiting_time": True})
                return {"response": ["📅 Great! What date and time works best for you? (e.g., 'tomorrow at 3 PM')"]}
            else:
                return {"response": ["❌ Please provide a valid Gmail address."]}

        #------------------ Step 3: Time collection -----------------------------------------------------------
        elif conversation_state["awaiting_time"]:
            parsed_time = dateparser.parse(
                user_input, settings={"TIMEZONE": "Asia/Kolkata", "RETURN_AS_TIMEZONE_AWARE": False}
            )
            if parsed_time:
                link = create_google_meet_invite(conversation_state["email"], parsed_time)
                send_meeting_email(conversation_state["email"], link)
                conversation_state.update({"awaiting_email": False, "awaiting_time": False, "email": ""})
                return {"response": [f"✅ Your demo has been booked!\nGoogle Meet Link: {link}"]}
            else:
                return {"response": ["❌ Sorry, I couldn't understand the date and time. Try again like '8 April at 4 PM'."]}

        #------------ Step 4: Fallback to generate_response (for queries like 'hi', 'hello') -----------------
        else:
            chatbot_reply = generate_response(user_input)
            return {"response": [chatbot_reply]}

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

#---------------------- Exception Handler for Validation Errors (Malformed JSON, Missing Fields) ----------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request format. Please check your input JSON."}
    )

#---------------------- Generic Exception Handler for Internal Errors --------------------------------------------------------
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Internal Server Error: {traceback.format_exc()}")  # Logs the full error traceback
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please try again later."}
    )


#----------------------------------- Allow FastAPI to Run Inside Jupyter/Colab ------------------------------------------------
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# In[ ]:




