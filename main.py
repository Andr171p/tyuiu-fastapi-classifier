from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api_v1.routers.classifier import classifier_router

from src.config import settings


app = FastAPI(
    title=settings.api_v1.name,
)

app.include_router(classifier_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
