from fastapi import FastAPI
from api.utils import load_inference_pipeline
from pydantic import BaseModel

from typing import Dict, Union, List

app = FastAPI()

PIPELINE = load_inference_pipeline(
    "./artifacts/trained_model", model_name="distilbert-base-multilingual-cased"
)


class InputText(BaseModel):
    text: str


@app.post(
    "/predict",
)
def predict(input_text: List[InputText]) -> List[Dict[str, Union[str, float]]]:
    return PIPELINE([i.text for i in input_text])
