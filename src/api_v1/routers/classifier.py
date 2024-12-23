from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.ml.pipeline.classifier import model_pipeline

from src.api_v1.schemas.applicant import (
    ApplicantSchema,
    ApplicantsSchema,
    ApplicantResponse
)

from src.config import settings


classifier_router = APIRouter(
    prefix=f"{settings.api_v1.prefix}/classifier",
    tags=["Probability prediction"]
)


@classifier_router.post(path="/predict/applicant/", response_model=ApplicantResponse)
async def predict_applicant(applicants: ApplicantSchema) -> JSONResponse:
    data = applicants.model_dump()
    prediction = model_pipeline.predict_proba(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': prediction[0][0]
        }
    )


@classifier_router.post(path="/predict/applicants/", response_model=...)
async def predict_applicants(applicants: ApplicantsSchema) -> JSONResponse:
    data = applicants.model_dump()['applicants']
    predictions = model_pipeline.predict_proba(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': predictions
        }
    )
