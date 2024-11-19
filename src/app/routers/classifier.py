from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.app.middlewares.globals import g
from src.app.schemas.user import UserSchema, UsersSchema
from src.app.schemas.response import UserResponseSchema, UsersResponseSchema


classifier_router = APIRouter(
    prefix="/classifier",
    tags=["Probability prediction"]
)


@classifier_router.post(path="/predict/", response_model=UserResponseSchema)
async def predict_user(user: UserSchema) -> JSONResponse:
    data = user.model_dump()
    model = g.model
    prediction = model.predict(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': prediction
        }
    )


@classifier_router.post(path="/predict_multy", response_model=UsersResponseSchema)
async def predict_users(users: UsersSchema) -> JSONResponse:
    data = users.model_dump()['data']
    model = g.model
    predictions = model.predict(data)
    return JSONResponse(
        content={
            'status': 'ok',
            'data': predictions
        }
    )
