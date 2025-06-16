import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH")
HF_SENTIMENT_TOKEN = os.getenv("HF_SENTIMENT_TOKEN")
SUMMARIZATION_MODEL_PATH = os.getenv("SUMMARIZATION_MODEL_PATH")
HF_SUMMARIZATION_TOKEN = os.getenv("HF_SUMMARIZATION_TOKEN")

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, token=HF_SENTIMENT_TOKEN)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, token=HF_SENTIMENT_TOKEN).to(device)

summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_PATH, token=HF_SUMMARIZATION_TOKEN)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_PATH, token=HF_SUMMARIZATION_TOKEN).to(device)
