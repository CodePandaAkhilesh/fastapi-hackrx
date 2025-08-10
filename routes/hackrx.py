import os
import requests
import hashlib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

router = APIRouter()

# Pinecone and Gemini setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class HackRxRequest(BaseModel):
    pdf_url: str
    questions: list[str]

def download_pdf(pdf_url):
    """Download PDF from URL and return file path."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        file_hash = hashlib.sha256(pdf_url.encode()).hexdigest()[:16]
        file_path = f"/tmp/{file_hash}.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

def process_pdf(file_path):
    """Load, split, and embed PDF into Pinecone."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
    return vectorstore

def answer_questions(vectorstore, questions):
    """Answer each question based on Pinecone search."""
    answers = []
    for q in questions:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(q)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {q}"
        resp = llm.invoke(prompt)
        answers.append(resp.content.strip())
    return answers

@router.post("/run")
async def run_hackrx(request: HackRxRequest):
    pdf_path = download_pdf(request.pdf_url)
    vectorstore = process_pdf(pdf_path)
    answers = answer_questions(vectorstore, request.questions)
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
