from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from typing import List, Dict
from model_load import HTRModelLoader
from stroke_processor import StrokeProcessor
import torch

app = FastAPI(title="Air Canvas HTR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the entire `web/public` directory so the viewer HTML and assets
# are available directly from the API server. This exposes paths like
# /avatar/avatar_viewer.html and /avatar/avatar.gltf (since `avatar` is a
# subfolder under web/public).
public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'public'))
app.mount('/', StaticFiles(directory=public_dir), name='public')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_loader = HTRModelLoader(
    checkpoint_path='../models/htr/checkpoint_finetuned.pth',
    device=device
)
stroke_processor = StrokeProcessor()

class Point(BaseModel):
    x: float
    y: float
    timestamp: float

class StrokeData(BaseModel):
    user_id: str
    strokes: List[List[Point]]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

@app.post("/recognize")
async def recognize_handwriting(data: StrokeData):
    try:
        strokes_dict = [
            [{'x': pt.x, 'y': pt.y} for pt in stroke]
            for stroke in data.strokes
        ]
        
        image = stroke_processor.strokes_to_image(strokes_dict)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid stroke data")
        
        input_tensor = stroke_processor.prepare_input(image)
        
        prediction = model_loader.predict(input_tensor)
        
        return {
            "user_id": data.user_id,
            "recognized_text": prediction,
            "confidence": 0.95
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
