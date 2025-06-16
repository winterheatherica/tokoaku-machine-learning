from flask import Blueprint, request, jsonify
import torch
from app.models.model_loader import sentiment_model, sentiment_tokenizer

sentiment_bp = Blueprint('sentiment', __name__)

@sentiment_bp.route('/analyze-sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "Teks kosong"}), 400

    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True).to(sentiment_model.device)
        outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_id].item()

        label_map = {0: "Negative", 1: "Positive"}
        label = label_map.get(predicted_id, f"Label-{predicted_id}")

        return jsonify({
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
