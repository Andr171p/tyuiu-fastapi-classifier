from typing import List, Literal

from pydantic import BaseModel


class ApplicantSchema(BaseModel):
    gender: Literal['М', 'Ж']
    gpa: float
    priority: int
    points: int
    direction: str


class ApplicantsSchema(BaseModel):
    applicants: List[ApplicantSchema]


class ApplicantPrediction(BaseModel):
    prediction: float


class ApplicantResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: ApplicantPrediction


class ApplicantsResponse(BaseModel):
    status: Literal['ok'] = 'ok'
    data: List[float]

