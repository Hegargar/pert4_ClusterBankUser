import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---------------------------------------------------------
# Path absolut ke folder tempat app.py berada
# Ini penting agar file pendukung (model, scaler, dll) selalu
# ditemukan, apa pun working directory yang dipakai Streamlit Cloud.
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")
PROFILE_PATH = os.path.join(BASE_DIR, "cluster_profile.csv")

# ---------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    layout="centered",
)

st.title("Segmentasi Nasabah Kartu Kredit")
st.markdown(
    """
    Aplikasi ini menggunakan model **K-Means Clustering** yang telah dilatih
    untuk mengelompokkan nasabah kartu kredit ke dalam beberapa segmen
    berdasarkan profil dan perilaku transaksinya.

    Masukkan data nasabah di bawah ini, lalu klik **Prediksi Segmen**
    untuk mengetahui nasabah tersebut termasuk cluster yang mana.
    """
)

# ---------------------------------------------------------
# Memuat model, scaler, daftar fitur, dan profil cluster
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File tidak ditemukan: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"File tidak ditemukan: {SCALER_PATH}")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error(f"File tidak ditemukan: {FEATURES_PATH}")
        st.stop()
    if not os.path.exists(PROFILE_PATH):
        st.error(f"File tidak ditemukan: {PROFILE_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    profile = pd.read_csv(PROFILE_PATH, index_col=0)
    return model, scaler, features, profile


model, scaler, features, cluster_profile = load_artifacts()

# ---------------------------------------------------------
# Form input data nasabah
# ---------------------------------------------------------
st.subheader("Input Data Nasabah")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Usia Nasabah (tahun)", min_value=18, max_value=100, value=40)
    months_on_book = st.number_input("Lama Menjadi Nasabah (bulan)", min_value=0, max_value=120, value=36)
    credit_limit = st.number_input("Limit Kredit (Credit Limit)", min_value=0.0, value=8000.0, step=100.0)
    total_revolving_bal = st.number_input("Saldo Bergulir (Total Revolving Balance)", min_value=0.0, value=1000.0, step=50.0)

with col2:
    total_trans_amt = st.number_input("Total Nominal Transaksi (Total Trans Amt)", min_value=0.0, value=4000.0, step=100.0)
    total_trans_ct = st.number_input("Total Jumlah Transaksi (Total Trans Ct)", min_value=0, value=60)
    total_relationship_count = st.number_input("Jumlah Produk Bank yang Dimiliki (Total Relationship Count)", min_value=1, max_value=10, value=4)
    avg_utilization_ratio = st.slider("Rasio Utilisasi Kartu (Avg Utilization Ratio)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

input_dict = {
    "Customer_Age": customer_age,
    "Months_on_book": months_on_book,
    "Credit_Limit": credit_limit,
    "Total_Revolving_Bal": total_revolving_bal,
    "Total_Trans_Amt": total_trans_amt,
    "Total_Trans_Ct": total_trans_ct,
    "Total_Relationship_Count": total_relationship_count,
    "Avg_Utilization_Ratio": avg_utilization_ratio,
}

# ---------------------------------------------------------
# Prediksi cluster
# ---------------------------------------------------------
if st.button("Prediksi Segmen", use_container_width=True):
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)

    cluster_pred = int(model.predict(input_scaled)[0])

    st.success(f"Nasabah ini termasuk ke dalam Cluster {cluster_pred}")

    st.subheader("Profil Rata-rata Cluster Ini")
    st.dataframe(cluster_profile.loc[[cluster_pred]])

    st.subheader("Perbandingan dengan Seluruh Cluster")
    st.dataframe(cluster_profile)

    st.info(
        """
        Catatan interpretasi umum (sesuaikan dengan hasil analisis pada notebook):
        - Cluster dengan Credit_Limit, Total_Trans_Amt, dan Total_Trans_Ct yang tinggi
          cenderung merupakan segmen nasabah premium / high-spender.
        - Cluster dengan Avg_Utilization_Ratio yang tinggi namun aktivitas transaksi lebih
          rendah cenderung merupakan segmen nasabah reguler yang lebih bergantung pada limit kreditnya.
        """
    )

st.markdown("---")
st.caption(
    "Model: K-Means Clustering. Dataset: Credit Card Customers (Kaggle). "
    "Dibuat untuk Tugas Mandiri Pertemuan 4 - Data Science Artificial Intelligence Course, Universitas Gunadarma"
)
