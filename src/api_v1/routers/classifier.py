from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.ml.pipeline.classifier import model_pipeline

from src.api_v1.schemas.applicant import (
    ApplicantSchema,
    ApplicantsSchema,
    ApplicantResponse,
    ApplicantsResponse
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
        status_code=status.HTTP_200_OK,
        content={
            "data": {
                "status": "ok",
                "prediction": float(prediction[0][1])
            }
        }
    )


@classifier_router.post(path="/predict/applicants/", response_model=ApplicantsResponse)
async def predict_applicants(applicants: ApplicantsSchema) -> JSONResponse:
    data = applicants.model_dump()
    predictions = model_pipeline.predict_proba(data['applicants'])
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "predictions": [
                float(prediction[1])
                for prediction in predictions
            ]
        }
    )
