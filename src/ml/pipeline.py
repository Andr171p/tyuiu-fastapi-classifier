from typing import List, Tuple
from sklearn.pipeline import Pipeline

from src.ml.pipes.imputer import Imputer
from src.ml.pipes.labels import LabelsImputer
from src.ml.pipes.ohe import OHE
from src.ml.pipes.scaler import Scaler
from src.ml.pipes.model import BinaryClassifierModel

features = ['education', 'study_form', 'reception_form', 'speciality']

columns = ['gender', 'hostel']

pipeline = Pipeline([
    ("binary_imputer", LabelsImputer()),
    ("one_hot_encoder", OHE(features=features)),
    ("standard_scaler", Scaler())
])


from src.app.schemas.user import UserSchema

user = UserSchema(
    gender='М',
    hostel='нет',
    gpa=4.5,
    priority=1,
    exams_points=220,
    bonus_points=10,
    education='Среднее общее образование',
    study_form='Очная',
    reception_form='Общий конкурс',
    speciality='01.03.02 Прикладная математика и информатика'
)
imp = Imputer()
print(user.model_dump())
df = imp.transform(user.model_dump())
print(df)
res = pipeline.transform(df)

model = BinaryClassifierModel()
predict = model.predict_proba(res)
print(predict)
