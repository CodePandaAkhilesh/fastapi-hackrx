import os
import requests
import hashlib
import time
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI router
router = APIRouter()

# Fetch API keys and config values from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Request model for the /run endpoint
class HackRxRequest(BaseModel):
    documents: str       # URL to PDF document
    questions: list[str] # List of questions to answer

@router.post("/run")
async def run_hackrx(payload: HackRxRequest):
    start_total = time.time()  # Track total execution time

    # Create a unique document ID using MD5 hash of the PDF URL
    doc_id = hashlib.md5(payload.documents.encode("utf-8")).hexdigest()

    # Initialize Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Download PDF file from provided URL
    pdf_path = "temp_doc.pdf"
    try:
        response = requests.get(payload.documents)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

    # Load PDF using LangChain's PyPDFLoader
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading PDF: {str(e)}")

    # Split PDF content into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Add doc_id metadata to each chunk for filtering during retrieval
    for i, d in enumerate(split_docs):
        d.metadata["doc_id"] = doc_id

    # Store document chunks in Pinecone vector store
    try:
        PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            ids=[f"{doc_id}-{i}" for i in range(len(split_docs))]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing vectors in Pinecone: {str(e)}")

    # Store a "marker vector" for the document itself (optional metadata)
    try:
        marker_embedding = embeddings.embed_query(payload.documents)
        index.upsert(vectors=[(doc_id, marker_embedding)])
    except Exception:
        pass  # Ignore marker embedding failure

    # Initialize Pinecone vector store wrapper for similarity search
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Initialize Google Generative AI LLM for answering questions
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    # Async function to answer a single question
    async def answer_question(question: str):
        start_q = time.time()
        try:
            # Perform similarity search in vector DB to get relevant context
            docs = await asyncio.to_thread(
                vectorstore.similarity_search, 
                question, 
                k=5, 
                filter={"doc_id": doc_id}
            )
            context = "\n\n".join([doc.page_content for doc in docs])

            # Create prompt for the LLM
            prompt = f"""Answer the following question strictly based on the provided context.
The answer must be concise but at least 10 words long.
If the answer is one word, explain it briefly.

Context:
{context}

Question: {question}

Answer:"""

            # Call the LLM with the prompt
            answer = await asyncio.to_thread(llm.invoke, prompt)
            return answer.content.strip(), time.time() - start_q
        except Exception as e:
            return f"Error during search: {str(e)}", time.time() - start_q

    # Run all question-answering tasks concurrently
    results = await asyncio.gather(*(answer_question(q) for q in payload.questions))

    # Extract answers and timing information
    final_answers = [r[0] for r in results]
    question_times = [r[1] for r in results]

    # Calculate performance metrics
    total_time = time.time() - start_total
    avg_time = sum(question_times) / len(question_times) if question_times else 0
    successful = sum(1 for a in final_answers if not a.lower().startswith("error"))
    accuracy = (successful / len(final_answers) * 100) if final_answers else 0

    # Print total processing time for debugging
    print(f"Total response time: {total_time:.2f} seconds")

    # Delete temporary PDF file after processing
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except Exception as e:
        print(f"Warning: Failed to delete temp PDF file: {str(e)}")

    # Return answers as JSON response
    return {"answers": final_answers}
