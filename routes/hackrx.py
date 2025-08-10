import os
import requests
import hashlib
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

router = APIRouter()

# --- Models ---
class HackRxRequest(BaseModel):
    pdf_url: str
    questions: list[str]

# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or GOOGLE_API_KEY in environment variables")

# --- Initialize Clients ---
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# --- Helper functions ---
def download_pdf(pdf_url, save_path):
    r = requests.get(pdf_url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    with open(save_path, "wb") as f:
        f.write(r.content)

def pdf_hash(pdf_url):
    return hashlib.sha256(pdf_url.encode()).hexdigest()

def index_exists(vectorstore, namespace):
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    return namespace in stats.get("namespaces", {})

def process_pdf(pdf_url, namespace):
    pdf_path = f"/tmp/{namespace}.pdf"
    download_pdf(pdf_url, pdf_path)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    return vectorstore

def query_pdf(vectorstore, questions):
    answers = []
    for q in questions:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""You are an expert assistant. 
        Use the following context to answer the question. 
        If the answer is not present in the context, say 'Answer not found in the document.'

        Context:
        {context}

        Question: {q}
        Answer:"""

        resp = llm.invoke(prompt)
        answers.append(resp.content.strip())

    return answers

# --- Routes ---
@router.post("/run")
async def run_hackrx(data: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    namespace = pdf_hash(data.pdf_url)
    vectorstore = PineconeVectorStore(
        index=pc.Index(INDEX_NAME),
        embedding=embeddings,
        namespace=namespace
    )

    if not index_exists(vectorstore, namespace):
        vectorstore = process_pdf(data.pdf_url, namespace)

    answers = query_pdf(vectorstore, data.questions)

    return {"answers": answers}


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
