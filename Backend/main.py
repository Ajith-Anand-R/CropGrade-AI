import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Import inference pipeline
# Ensure parent directory is in path so we can import Models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.inference import predict_batch

app = FastAPI(
    title="CropGrade AI API",
    description="Two-stage tomato quality grading and freshness classification API",
    version="1.0.0"
)

# Enable CORS for frontend web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "healthy", "message": "CropGrade AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file extension
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG and PNG are supported."
        )

    # Save to a temporary file
    try:
        suffix = extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Execute 2-stage inference pipeline
        results = predict_batch(temp_path)

        # Cleanup temp file
        os.remove(temp_path)

        return results

    except Exception as e:
        # Ensure temp file is cleaned up in case of error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
