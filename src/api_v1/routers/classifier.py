from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.ml import ml_pipeline

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
    prediction = ml_pipeline.predict(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': prediction[0][0]
        }
    )


@classifier_router.post(path="/predict/applicants/", response_model=UsersResponseSchema)
async def predict_users(users: UsersSchema) -> JSONResponse:
    data = users.model_dump()['data']
    predictions = ml_pipeline.predict(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': predictions
        }
    )
