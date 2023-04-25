from transformers.pipelines.base import Pipeline
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def load_inference_pipeline(traine_model_path: str, model_name: str) -> Pipeline:
    model = AutoModelForSequenceClassification.from_pretrained(traine_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
