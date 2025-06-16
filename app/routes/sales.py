from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import math
from datetime import timedelta
from xgboost import XGBRegressor
import os

from app.utils.forecast_utils import (
    WINDOW, FORECAST_STEP,
    create_windows, forecast_future,
    format_rupiah, get_gemini_analysis
)

sales_bp = Blueprint('sales', __name__)

@sales_bp.route('/predict-sales', methods=['POST'])
def predict_sales():
    try:
        data = request.get_json()
        if not data or 'sales' not in data:
            return jsonify({'error': 'Data JSON tidak valid atau tidak mengandung "sales"'}), 400

        df = pd.DataFrame(data['sales'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        if len(df) < WINDOW:
            return jsonify({
                'error': f'Data historis kurang dari {WINDOW} hari',
                'analysis': f'Tidak dapat membuat prediksi karena data historis kurang dari {WINDOW} hari'
            }), 400

        df['total_sales'] = df['total_sales'].apply(lambda x: np.log1p(x))
        values = df[['total_sales']].values

        X, y = create_windows(values, WINDOW)

        print("DEBUG - X shape:", X.shape)
        print("DEBUG - y shape:", y.shape)

        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            min_child_weight=10,
            gamma=0
        )
        model.fit(X, y)

        last_window = values[-WINDOW:, 0]
        future_preds_log = forecast_future(model, last_window, FORECAST_STEP)
        future_preds = np.expm1(future_preds_log)

        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_STEP)
        predictions_dict = {
            str(date.date()): {"total_sales": format_rupiah(float(sales))}
            for date, sales in zip(future_dates, future_preds)
        }

        historical_data = df.tail(WINDOW).copy()
        historical_data['total_sales'] = historical_data['total_sales'].apply(lambda x: np.expm1(x))

        gemini_analysis = get_gemini_analysis(historical_data, predictions_dict)

        print("DEBUG - Predictions:\n", predictions_dict)

        return jsonify({
            "predictions": predictions_dict,
            "analysis": gemini_analysis
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500
