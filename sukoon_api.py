from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated
import uvicorn

# from fastapi import Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import json
# from datetime import datetime

from sukoon import chat

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sukoon", description="API for the Sukoon mental health support system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SukoonRequest(BaseModel):
    input: str

class SukoonResponse(BaseModel):
    output: str

@app.post("/query", response_model = SukoonResponse)
async def process_query(request: SukoonRequest):
    config = {"configurable": {"thread_id":"1"}}
    user_input = request.input
    response = chat(user_input, config)
    return SukoonResponse(output = response.content)
    
@app.get("/")
async def root():
    return {"message": "Welcome to the Sukoon API. Use the /query endpoint to interact with the system."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8001)

# for google analytics
# templates = Jinja2Templates(directory="templates")

# # Track conversation events
# async def track_conversation_event(conversation_id: str, event_type: str, data: dict):
#     # Send to Google Analytics
#     event = {
#         'conversation_id': conversation_id,
#         'event_type': event_type,
#         'timestamp': datetime.utcnow().isoformat(),
#         'data': data
#     }
#     # Log for analysis
#     print(json.dumps(event))

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     # Your existing chat logic
#     conversation_id = "unique_id"  # Generate unique ID
    
#     # Track conversation start
#     await track_conversation_event(
#         conversation_id=conversation_id,
#         event_type="conversation_start",
#         data={
#             'user_id': request.client.host,
#             'timestamp': datetime.utcnow().isoformat()
#         }
#     )