from fastapi import FastAPI
from routes import hackrx  # Import the HackRx route handler

# Create FastAPI application instance
app = FastAPI()

# Include the HackRx router under the "/hackrx" URL prefix
# All endpoints in hackrx.py will be accessible at /hackrx/<endpoint>
app.include_router(hackrx.router, prefix="/hackrx", tags=["HackRx"])

# Root endpoint to check if the API is running
@app.get("/")
async def root():
    return {"message": "HackRx API is running"}  # Simple health check response
