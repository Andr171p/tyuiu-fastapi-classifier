from typing import Literal
from pydantic import BaseModel, field_validator


class UserSchema(BaseModel):
    gender: Literal['М', 'Ж']
    hostel: Literal['да', 'нет']
    gpa: float
    priority: int
    exams_points: int
    bonus_points: int
    education: str
    study_form: str
    reception_form: Literal['Очная', 'Заочная', 'Очно-Заочная']
    speciality: str

    @field_validator('gpa')
    @classmethod
    def validate_gpa(cls, v: float) -> float:
        if v < 3 or v > 5:
            raise ValueError("Field `gpa` must be in range [3;5]")
        return v

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("Field `priority` must be in range [1;5]")
        return v

    @field_validator('exams_points')
    @classmethod
    def validate_exams_points(cls, v: int) -> int:
        if v < 0 or v > 310:
            raise ValueError("Field `exams_points` must be in range [0;310]")
        return v

    @field_validator('bonus_points')
    @classmethod
    def validate_bonus_points(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("Field `bonus_points` must be in range [0;10]")
        return v
