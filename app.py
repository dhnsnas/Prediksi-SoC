import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Judul Aplikasi
st.title("Prediksi SoC Baterai VRLA dengan SVR")
st.markdown("Aplikasi ini menggunakan model Support Vector Regression (SVR) untuk memprediksi State of Charge (SoC) berdasarkan input parameter baterai.")

# Load model
model_path = "model_svr.pkl"
df_path = "df.pkl"

if os.path.exists(model_path) and os.path.exists(df_path):
    svr_model = joblib.load(model_path)
    df = pd.read_pickle(df_path)
else:
    st.error("❌ File model atau data tidak ditemukan. Pastikan 'model_svr.pkl' dan 'df.pkl' tersedia di folder.")
    st.stop()

# Sidebar input user
st.sidebar.header("Input Parameter")

voltase = st.sidebar.number_input("Voltase [V]", min_value=0.0, value=100.0)
arus = st.sidebar.number_input("Arus [A]", min_value=0.0, value=40.7)
resistansi = st.sidebar.number_input("Resistansi [Ohm]", min_value=0.0, value=2.46)
waktu_dalam_detik = st.sidebar.number_input("Waktu [detik]", min_value=0, value=13250)

# Tombol prediksi
if st.sidebar.button("Prediksi SoC"):
    # Prediksi SoC
    input_data = [[float(voltase), float(arus), float(resistansi), float(waktu_dalam_detik)]]
    y_pred = svr_model.predict(input_data)
    st.success(f"Hasil Prediksi SoC: **{round(y_pred[0], 2)}%**")

# Tampilkan data (opsional)
with st.expander("📊 Lihat Data Discharge"):
    st.dataframe(df)
