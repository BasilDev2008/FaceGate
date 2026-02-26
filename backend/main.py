from fastapi import FastAPI  # main framework for our API
from fastapi.middleware.cors import CORSMiddleware  # allows frontend to talk to backend
from routes.register import router as register_router  # import register routes
app = FastAPI(
    title = "FaceGate",
    description= "Machine Learning face and voice recognition security system",
    version = "1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
app.include_router(register_router, prefix = "/api")
@app.get("/")
async def root():
    return {"message": "FaceGate is running"}
@app.get("/health")
async def health():
    return {"status": "healthy"}
