from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from database.engine import Base, engine, get_db
from database.models import Analysis
from config import Settings
from model import SentimentModel
from predict import predict_sentiment_farsi
from fastapi.middleware.cors import CORSMiddleware

import os
import re
import json
import copy
import collections
import numpy as np

import torch
from transformers import AutoTokenizer, BertConfig

# Setting Up Model & Tokenizer
print("Loading model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained(Settings.MODEL_NAME)
pt_model = SentimentModel(config)
pt_model = torch.load(Settings.MODEL_SAVE2, map_location=device)
print("Model loaded successfully.")

print("Loading tokenizer...")
path = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
tokenizer = AutoTokenizer.from_pretrained(path)
print("Tokenizer loaded successfully.")

# Initializing FastAPI
app = FastAPI()

# Setting Up Templates
templates = Jinja2Templates(directory="templates")

# Configuring CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Database Initialization on Startup
# @app.on_event("startup")
# async def init_tables():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

@app.post("/process-text/")
async def process_text(text: str = Form(...)):
    predicted_class, probabilities = predict_sentiment_farsi(pt_model, tokenizer, text, device)
    
    label_mapping = {0: "negative", 1: "positive"}

    # Flatten the probabilities (just in case it's nested)
    flat_probs = np.array(probabilities).flatten()

    # Optionally round for cleaner output
    formatted_probs = [round(float(prob), 6) for prob in flat_probs]

    return {"text": text, 
            "sentiment": label_mapping[predicted_class], 
            "accuracy": formatted_probs
            # "accuracy": probabilities.cpu().numpy().tolist()
            }


# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/process-text/", response_class=HTMLResponse)
# async def process_text(request: Request, text: str = Form(...), db: AsyncSession = Depends(get_db)):
    
#     # Running Sentiment Analysis
#     predicted_class, probabilities = predict_sentiment_farsi(pt_model, tokenizer, text, device)
    
#     # Formatting Output
#     probabilities_list = probabilities.cpu().numpy().tolist()
#     probabilities_json = json.dumps(probabilities_list)
    
#     label_mapping = {0: "negetive", 1: "positive"}
#     new_record = Analysis(
#         text=text,
#         result=label_mapping[predicted_class],
#         accuracy=probabilities_json
#     )
    
#     # Saving Data to Database
#     async with db.begin():
#         db.add(new_record)
#         await db.flush()
    
#     # Returning Response
#     return templates.TemplateResponse("index.html",                                      
#     {
#         "request": request,
#         "result": label_mapping[predicted_class],
#         "accuracy": probabilities
#     })

# Running the Application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



