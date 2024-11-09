import socketio
import torch
from fastapi import FastAPI, Request
from model import FitnessLSTMModel 
from model import FitnessClassifier as classifier
from transformers import pipeline

# Socket.io Server
sio = socketio.AsyncServer(async_mode='asgi')
app = FastAPI()
app.mount("/ws", socketio.ASGIApp(sio))

# Load models
input_size = 10
hidden_size = 128
num_layers = 2
output_size = 1
model = FitnessLSTMModel(input_size, hidden_size, num_layers, output_size)
nlp_model = pipeline("text-classification", model="bert-base-uncased")

# Event handlers for real-time communication
@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def analyze_activity(sid, data):
    activity_data = torch.tensor(data['activity']).unsqueeze(0)
    result = model(activity_data)
    feedback = data.get('feedback', 'No feedback')
    sentiment = nlp_model(feedback)[0]
    await sio.emit('activity_response', {'prediction': result.item(), 'sentiment': sentiment})

async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    prediction = classifier.predict(text)
    return {"prediction": prediction}

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

# Run with: uvicorn main:app --reload