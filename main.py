import uvicorn
from fastapi import FastAPI

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
