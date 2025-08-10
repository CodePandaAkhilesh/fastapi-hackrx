# routes/hackrx.py

import os
import requests
import hashlib
import tempfile
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or PINECONE_API_KEY in environment variables.")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

router = APIRouter()

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: list[str]

@router.post("/run")
async def run_hackrx(
    body: HackRxRequest,
    authorization: str = Header(None)
):
    # Simple Bearer token auth check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = authorization.split(" ")[1]
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if expected_token and token != expected_token:
        raise HTTPException(status_code=403, detail="Forbidden")

    pdf_url = body.documents
    questions = body.questions

    if not pdf_url.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    # Create a deterministic ID for the document (SHA256 of URL)
    doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()

    # Check if already in Pinecone
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) > 0:
        if doc_id in stats.get("namespaces", {}):
            # Skip re-upload, only query
            return await query_pdf(doc_id, questions)

    # Download PDF to temp file
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch PDF: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(response.content)
        tmp_pdf_path = tmp_pdf.name

    # Load and split
    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Embed and store in Pinecone
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=doc_id
    )

    # Query
    return await query_pdf(doc_id, questions)

async def query_pdf(doc_id: str, questions: list[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=doc_id
    )

    results = {}
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

    for q in questions:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {q}"
        answer = llm.invoke(prompt).content
        results[q] = answer

    return {"answers": results}

# import os
# import requests
# import hashlib
# import time
# import asyncio
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pinecone import Pinecone

# load_dotenv()

# router = APIRouter()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)

# class HackRxRequest(BaseModel):
#     documents: str       # URL to PDF
#     questions: list[str] # List of questions


# @router.post("/run")
# async def run_hackrx(payload: HackRxRequest):
#     start_total = time.time()

#     doc_id = hashlib.md5(payload.documents.encode("utf-8")).hexdigest()
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GEMINI_API_KEY
#     )

#     # Always download PDF, embed, and upload chunks (no existence check)
#     pdf_path = "temp_doc.pdf"
#     try:
#         response = requests.get(payload.documents)
#         response.raise_for_status()
#         with open(pdf_path, "wb") as f:
#             f.write(response.content)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

#     try:
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading PDF: {str(e)}")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_docs = text_splitter.split_documents(docs)

#     for i, d in enumerate(split_docs):
#         d.metadata["doc_id"] = doc_id

#     try:
#         PineconeVectorStore.from_documents(
#             documents=split_docs,
#             embedding=embeddings,
#             index_name=PINECONE_INDEX_NAME,
#             ids=[f"{doc_id}-{i}" for i in range(len(split_docs))]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error storing vectors in Pinecone: {str(e)}")

#     try:
#         marker_embedding = embeddings.embed_query(payload.documents)
#         index.upsert(vectors=[(doc_id, marker_embedding)])
#     except Exception:
#         pass

#     vectorstore = PineconeVectorStore(
#         index_name=PINECONE_INDEX_NAME,
#         embedding=embeddings
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0,
#         google_api_key=GEMINI_API_KEY
#     )

#     async def answer_question(question: str):
#         start_q = time.time()
#         try:
#             docs = await asyncio.to_thread(vectorstore.similarity_search, question, k=3, filter={"doc_id": doc_id})
#             context = "\n\n".join([doc.page_content for doc in docs])
#             prompt = f"""Answer the following question strictly based on the provided context.
# The answer must be concise but at least 10 words long.
# If the answer is one word, explain it briefly.

# Context:
# {context}

# Question: {question}

# Answer:"""
#             answer = await asyncio.to_thread(llm.invoke, prompt)
#             return answer.content.strip(), time.time() - start_q
#         except Exception as e:
#             return f"Error during search: {str(e)}", time.time() - start_q

#     # Run all questions concurrently
#     results = await asyncio.gather(*(answer_question(q) for q in payload.questions))

#     final_answers = [r[0] for r in results]
#     question_times = [r[1] for r in results]

#     total_time = time.time() - start_total
#     avg_time = sum(question_times) / len(question_times) if question_times else 0

#     successful = sum(1 for a in final_answers if not a.lower().startswith("error"))
#     accuracy = (successful / len(final_answers) * 100) if final_answers else 0

#     print(f"Total response time: {total_time:.2f} seconds")

#     # Delete the temp PDF file after processing is complete
#     try:
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)
#     except Exception as e:
#         print(f"Warning: Failed to delete temp PDF file: {str(e)}")

#     return {"answers": final_answers}
