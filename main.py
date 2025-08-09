from fastapi import FastAPI
from routes import hackrx

app = FastAPI()

app.include_router(hackrx.router, prefix="/hackrx", tags=["HackRx"])

@app.get("/")
async def root():
    return {"message": "HackRx API is running"}
