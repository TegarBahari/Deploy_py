import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Deteksi Dini Malaria",
    page_icon="🦟",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_malaria.pkl')
        return model
    except:
        return None

model = load_model()

st.title("🦟 Sistem Prediksi Malaria")
st.write("""
Website ini menggunakan metode **Klasifikasi (Random Forest)** untuk memprediksi 
kemungkinan seseorang terjangkit malaria berdasarkan gejala klinis awal.
""")
st.markdown("---")

st.sidebar.header("Data Gejala Pasien")
st.sidebar.write("Silakan isi sesuai kondisi saat ini:")

def ambil_input_user():
    demam = st.sidebar.selectbox('Apakah mengalami Demam Tinggi?', ('Tidak', 'Ya'))
    kepala = st.sidebar.selectbox('Apakah mengalami Sakit Kepala?', ('Tidak', 'Ya'))
    mual = st.sidebar.selectbox('Apakah mengalami Mual/Muntah?', ('Tidak', 'Ya'))
    menggigil = st.sidebar.selectbox('Apakah mengalami Menggigil?', ('Tidak', 'Ya'))
    otot = st.sidebar.selectbox('Apakah mengalami Nyeri Otot?', ('Tidak', 'Ya'))

    data_user = {
        'Demam_Tinggi': 1 if demam == 'Ya' else 0,
        'Sakit_Kepala': 1 if kepala == 'Ya' else 0,
        'Mual_Muntah': 1 if mual == 'Ya' else 0,
        'Menggigil': 1 if menggigil == 'Ya' else 0,
        'Nyeri_Otot': 1 if otot == 'Ya' else 0
    }
    
    fitur = pd.DataFrame(data_user, index=[0])
    return fitur

input_df = ambil_input_user()

st.subheader("Gejala yang Dimasukkan:")
st.dataframe(input_df)

if st.button('🔍 Analisis Sekarang'):
    if model:
        prediksi = model.predict(input_df)
        probabilitas = model.predict_proba(input_df)
        
        st.markdown("---")
        st.subheader("Hasil Analisis:")
        
        if prediksi[0] == 1:
            st.error("⚠️ **TERINDIKASI MALARIA**")
            st.write(f"Sistem mendeteksi kecocokan gejala sebesar **{probabilitas[0][1]*100:.2f}%**")
            st.warning("Saran: Segera kunjungi Puskesmas atau Rumah Sakit terdekat untuk tes darah.")
        else:
            st.success("✅ **KEMUNGKINAN SEHAT / NEGATIF**")
            st.write(f"Peluang sehat sebesar **{probabilitas[0][0]*100:.2f}%**")
            st.info("Saran: Tetap jaga kebersihan lingkungan dan gunakan kelambu saat tidur.")
    else:
        st.error("Error: Model tidak ditemukan. Harap jalankan 'train_model.py' terlebih dahulu.")

st.markdown("---")
st.caption("Tugas Web 2 | Dibuat dengan Python & Streamlit")