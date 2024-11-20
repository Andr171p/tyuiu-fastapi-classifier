import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.application import lifespan
from src.app.middlewares.globals import GlobalMiddleware
from src.app.routers.classifier import classifier_router
from src.config import settings


app = FastAPI(
    title=settings.app.name,
    lifespan=lifespan
)

app.include_router(classifier_router)

app.add_middleware(GlobalMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
