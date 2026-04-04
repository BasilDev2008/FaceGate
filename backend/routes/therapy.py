from fastapi import APIRouter, HTTPException  # for creating routes and handling errors
from pydantic import BaseModel  # for defining request data
from therapy_engine import TherapyEngine  # our therapy engine

router = APIRouter()  # create router

class StartSessionRequest(BaseModel):  # defines what data we expect
    user_id: str  # who is starting the session

@router.post("/therapy/start")  # POST request to start a therapy session
async def start_session(request: StartSessionRequest):
    try:
        engine = TherapyEngine()  # create therapy engine instance
        conversation_history = []  # empty conversation history to start
        engine.chat(request.user_id, conversation_history)  # start the session
        return {"message": "Session completed successfully"}  # return success
    except Exception as e:  # if anything goes wrong
        raise HTTPException(status_code=500, detail=str(e))  # return error