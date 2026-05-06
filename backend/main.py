from dotenv import load_dotenv  # loads environment variables
load_dotenv()  # reads .env file

from fastapi import FastAPI  # main framework for our API
from fastapi.middleware.cors import CORSMiddleware  # allows frontend to talk to backend
from routes.register import router as register_router
from routes.stream import router as stream_router  # import stream routes
from routes.therapy import router as therapy_router  # import therapy routes
from routes.recognize import router as recognize_router
app = FastAPI(  # create the app first
    title="FaceGate",
    description="AI powered face and voice recognition security system",
    version="1.0.0"
)
app.include_router(recognize_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register_router, prefix="/api")  # then add routers
app.include_router(stream_router, prefix="/api")
app.include_router(therapy_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "FaceGate is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}