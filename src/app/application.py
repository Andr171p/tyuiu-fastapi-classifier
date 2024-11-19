from fastapi import FastAPI
from contextlib import AbstractAsyncContextManager

from src.app.middlewares.globals import g
from src.model.pipeline import ModelPipeline


async def lifespan(app: FastAPI) -> AbstractAsyncContextManager[None]:
    model = ModelPipeline()
    model.create_pipeline()
    g.set_default("model", model)
    yield
    del model
