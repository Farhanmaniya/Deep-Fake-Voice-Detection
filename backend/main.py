from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.websocket_handler import router as websocket_router
from backend.api.metrics_handler import router as metrics_router
from backend.api.health_handler import router as health_router
from backend.config.settings import settings
import uvicorn

app = FastAPI(title="DeepFake Detection Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(websocket_router)
app.include_router(metrics_router)
app.include_router(health_router)

@app.get("/")
async def root():
    return {"message": "DeepFake Detection WebSocket Server Running"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=True)
