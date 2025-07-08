from flask import Blueprint, request, jsonify
from app.models.model_loader import summarization_model, summarization_tokenizer
import torch
import time

summarize_bp = Blueprint('summarize', __name__)

@summarize_bp.route('/create-summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    reviews = data.get('reviews', [])

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "Harap kirimkan list review dengan key 'reviews'"}), 400

    try:
        joined_text = ". ".join(reviews) + "."

        # Tokenisasi input
        input_tokens = summarization_tokenizer.encode(
            f"ringkaslah: {joined_text}",
            return_tensors="pt",
            max_length=1400,
            truncation=True
        ).to(summarization_model.device)

        input_length = input_tokens.shape[1]


        start = time.time()

        summary_ids = summarization_model.generate(
            input_tokens,
            min_length=100,
            max_length=270,
            num_beams=4,
            no_repeat_ngram_size=2,
            repetition_penalty=2.0,
            length_penalty=1.9,
            early_stopping=True
        )

        end = time.time()

        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({
            "summary": summary,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500