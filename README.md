# 📄 LLM-Powered Intelligent Query–Retrieval System

## 💻 GitHub Repository
**Code:** [https://github.com/CodePandaAkhilesh/fastapi-hackrx](https://github.com/CodePandaAkhilesh/fastapi-hackrx)  
**Demo API:** [https://fastapi-hackrx-1.onrender.com/hackrx/run](https://fastapi-hackrx-1.onrender.com/hackrx/run)

---

## 🚀 Problem Statement
Real-world teams in **insurance, HR, legal, and compliance** face challenges finding **clause-level answers** in long policies and contracts.  
Our solution:  
- Parse **PDF/DOCX/email** documents.  
- Retrieve **exact clauses** relevant to a natural language query.  
- Provide **explainable decisions** with traceable metadata.  
- Output answers as **structured JSON** for system integration.

---

## 📌 Key Features
✅ **Multi-format ingestion** – PDF, DOCX, email  
✅ **Large document scalability** – Handles hundreds of pages efficiently  
✅ **Clause-level evidence retrieval** – Traceable to page & section  
✅ **Semantic search** – Uses Gemini embeddings + Pinecone  
✅ **Strict context answering** – Reduces hallucinations  
✅ **Structured JSON output** – Easy to integrate with existing systems  
✅ **Parallel Q&A** – Handles multiple questions in a single request  

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|------------|
| Backend API | FastAPI |
| LLM | Google Gemini (gemini-2.5-flash) |
| Embeddings | Gemini `models/embedding-001` |
| Vector DB | Pinecone |
| PDF Parsing | pypdf + LangChain PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Deployment | Render |
| Utilities | python-dotenv, requests, asyncio |

---

## ⚙️ Architecture

### **Document Flow**
1. **Ingestion** – PDF/DOCX/email via URL or file upload.  
2. **Text Preprocessing** – Chunk into 1000 characters, 200-character overlap.  
3. **Embedding Generation** – Gemini embeddings for semantic representation.  
4. **Vector Storage** – Store in Pinecone with `doc_id` for filtering.

### **Query Flow**
1. User submits natural language query.  
2. Pinecone retrieves top-k relevant chunks (filtered by `doc_id`).  
3. Gemini LLM answers **only** from retrieved context.  
4. Output returned as **JSON with answer & evidence metadata**.

---

## 📂 Project Structure
.
├── .env                     # Environment variables (API keys, config)
├── main.py                  # FastAPI entry point
├── requirements.txt         # Dependencies list
├── routes/
│   ├── __pycache__/         # Compiled Python files (auto-generated)
│   └── hackrx.py            # Core logic for document ingestion, embedding, Q&A
├── .venv/                   # Virtual environment (local to project)
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── pyvenv.cfg
└── __pycache__/             # Compiled Python cache for main.py

## 🧑‍💻 Installation & Setup

### Clone Repository
```bash

git clone https://github.com/CodePandaAkhilesh/fastapi-hackrx.git
cd fastapi-hackrx


## **Install Dependencies**

pip install -r requirements.txt


## Environment Variables

## create .env file:

GEMINI_API_KEY = your_gemini_api_key
PINECONE_API_KEY= your_pinecone_api_key
PINECONE_ENVIRONMENT= us-east-1 (default)
PINECONE_INDEX_NAME = your Index Name


▶️ Running the App
Development Mode
uvicorn main:app --reload



Required API Structure:
Request Format:

POST https://fastapi-hackrx-1.onrender.com/hackrx/run
Content-Type: application/json

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
Response Format:

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}



💡 Implementation Highlights

Document Hashing: MD5 to uniquely tag docs (doc_id) for filtering.

Chunking: 1000 chars, 200 overlap → balances context & redundancy.

Marker Vector: Stores doc-level embedding for quick reference.

Concurrency: asyncio.gather for parallel Q&A handling.

Cleanup: Temporary files deleted post-processing.



🏆 Unique Selling Points (USP)

Strict Context Reliance – Answers only from retrieved chunks → minimizes hallucinations.

Traceable Evidence – Chunk metadata maps back to original doc page & section.

Multi-domain Ready – Optimized for insurance, legal, HR, compliance.

Parallel Processing – Handles multiple questions instantly.

Scalable Deployment – Runs seamlessly on Render with managed Pinecone.



📈 Business Impact

Insurance – Rapid claims triage & eligibility checks.

Legal – Contract clause extraction & due diligence.

HR – Policy Q&A for employee benefits & leave rules.

Compliance – Regulatory document search with auditable evidence.



🔮 Future Enhancements

Cross-document clause matching.

Web UI for upload, search, and clause highlighting.

Role-based access control & audit logging.

Local FAISS fallback for offline/low-cost scenarios.

Incremental indexing (no full re-embedding needed).



📞 Contact

Akhilesh Verma – av14021999@gmail.com

Krishnakant Kushwaha – kushwahakrishnakant979@gmail.com



Test API: https://fastapi-hackrx-1.onrender.com/hackrx/run
GitHub: https://github.com/CodePandaAkhilesh/fastapi-hackrx


📜 License
MIT License
