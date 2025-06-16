import os
import math
import numpy as np
import pandas as pd
from datetime import timedelta
import google.generativeai as genai

WINDOW = 30
FORECAST_STEP = 30

def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

def forecast_future(model, last_window, steps):
    preds = []
    window = last_window.copy()
    for _ in range(steps):
        p = model.predict(window.reshape(1, -1))[0]
        preds.append(p)
        window = np.roll(window, -1)
        window[-1] = p
    return np.array(preds)

def format_rupiah(value):
    return "Rp. {:,.0f}".format(value).replace(",", ".")

def get_gemini_analysis(historical_df, predictions_json):
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        return "Analisis Gemini tidak tersedia: API Key belum disetel"

    genai.configure(api_key=api_key)

    historical_md = historical_df.reset_index()[['date', 'total_sales']] \
        .rename(columns={'total_sales': 'actual_sales'}) \
        .to_markdown(index=False)

    future_dates = list(predictions_json.keys())
    future_sales = [item['total_sales'] for item in predictions_json.values()]
    future_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_sales})
    future_md = future_df.to_markdown(index=False)

    data_str = f"=== Data Historis ===\n{historical_md}\n\n=== Prediksi 30 Hari ===\n{future_md}"

    prompt = f"""Anda adalah analis bisnis e-commerce profesional. *JANGAN BANYAK BASA BASI. tolong buat paragraf singkat/ringkas saja, buat dalam satu paragraf seperti contoh output. ingat anda berbicara kepada pedagang bukan developer model !*

    Tugas Anda:
    - Jelaskan tren penjualan (naik/turun/stabil) dalam 30 hari ke depan.
    - Berikan alasan prediksi tersebut (misalnya: libur Lebaran/libur sekolah, akhir pekan, hari kerja, tanggal cantik seperti 11.11, dll).
    - Berikan saran strategi bisnis kepada pedagang e-commerce agar bisa memanfaatkan momen atau mengatasi penurunan penjualan.
    - Gunakan Bahasa Indonesia yang profesional dan mudah dimengerti.
    - markdown analisis penting yang harus di highlight oleh seller

    ini adalah tanggal penting :
    # tanggal cantik promo :
    "2025-01-01", "2025-01-11", "2025-01-25",
    "2025-02-02", "2025-02-14", "2025-02-25",
    "2025-03-03", "2025-03-08", "2025-03-25",
    "2025-04-04", "2025-04-25",
    "2025-05-05", "2025-05-25",
    "2025-06-06", "2025-06-25",
    "2025-07-07", "2025-07-25",
    "2025-08-08", "2025-08-17", "2025-08-25",
    "2025-09-09", "2025-09-25",
    "2025-10-10", "2025-10-25",
    "2025-11-11", "2025-11-25",
    "2025-12-12", "2025-12-25", "2025-12-31"
    # 2025 Holidays : 
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-03-31",
    "2025-04-18", "2025-04-20", "2025-05-01", "2025-05-12", "2025-05-29",
    "2025-06-01", "2025-06-06", "2025-06-27", "2025-08-17", "2025-09-05",
    "2025-12-25"
    # Ramadan periods (1 month before Lebaran) : ("2025-03-01", "2025-03-30")
    # Lebaran periods (1 week after Idul Fitri) : ("2025-03-31", "2025-04-07")
    # School holiday periods : ("2025-06-15", "2025-07-15"),("2025-12-15", "2026-01-05")

    *Contoh output:*
    Berdasarkan tren penjualan sebelumnya, diprediksi akan terjadi penurunan penjualan setelah libur Lebaran karena masyarakat kembali fokus bekerja. Disarankan untuk membuat promo pasca-Lebaran agar menarik perhatian pembeli. Perlu diingat bahwa kondisi pasar bisa berubah sewaktu-waktu, maka strategi ini hanya sebagai acuan awal.
    
    Berikut data penjualan (30 hari terakhir dan 30 hari ke depan):
    {data_str}"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gagal mendapatkan analisis Gemini: {str(e)}"