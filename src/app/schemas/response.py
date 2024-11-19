from pydantic import BaseModel
from typing import List, Literal


class UserResponseSchema(BaseModel):
    status: Literal['ok'] = 'ok'
    data: float


class UsersResponseSchema(BaseModel):
    status: Literal['ok'] = 'ok'
    data: List[float]
