import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Rekomendasi HP", page_icon="📱", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>📱 Rekomendasi HP Pintar</h1>
    <p style='text-align:center; font-size:18px;'>
        Masukkan budget, RAM minimal, dan resolusi kamera untuk menemukan HP terbaik sesuai kebutuhanmu.
    </p>
    <hr>
""", unsafe_allow_html=True)

model = joblib.load("model_rekomendasi_hp.pkl")
label_encoder = joblib.load("label_encoder.pkl")
df = pd.read_csv("dataset.handphone.csv")

if "Resolusi_kamera_num" not in df.columns:
    df["Resolusi_kamera_num"] = df["Resolusi_kamera"].str.replace("MP", "").astype(int)

st.sidebar.header("🔍 Filter Pencarian")
budget = st.sidebar.number_input("Budget Maksimal (Rp)", min_value=500000, step=500000)
ram_min = st.sidebar.number_input("RAM Minimal (GB)", min_value=1, step=1)
kamera_min = st.sidebar.number_input("Resolusi Kamera Minimal (MP)", min_value=1, step=1)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

if st.sidebar.button("Cari Rekomendasi"):
    hasil = df[
        (df["Harga"] <= budget) &
        (df["Ram"] >= ram_min) &
        (df["Resolusi_kamera_num"] >= kamera_min)
    ]

    if hasil.empty:
        st.error("❌ Tidak ada HP yang cocok dengan kriteria kamu.")
    else:
        fitur_model = hasil[[
            "Harga",
            "Ram",
            "Memori_internal",
            "Ukuran_layar",
            "Resolusi_kamera_num",
            "Kapasitas_baterai",
            "Rating_pengguna"
        ]]

        prediksi = model.predict(fitur_model)
        hasil["Prediksi_Kelas"] = label_encoder.inverse_transform(prediksi)

        st.success("🎉 Rekomendasi HP berhasil ditemukan!")

        st.markdown("<h3>📋 Hasil Rekomendasi</h3><hr>", unsafe_allow_html=True)

        for _, row in hasil.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://via.placeholder.com/250?text=" + row["Nama_hp"].replace(" ", "+"))
                with col2:
                    st.markdown(f"### {row['Nama_hp']}")
                    st.markdown(f"**Brand:** {row['Brand']}")
                    st.markdown(f"**Harga:** Rp {row['Harga']:,}")
                    st.markdown(f"**RAM:** {row['Ram']} GB")
                    st.markdown(f"**Kamera:** {row['Resolusi_kamera']}")
                    st.markdown(f"**Baterai:** {row['Kapasitas_baterai']} mAh")
                    st.markdown(f"**Rating Pengguna:** ⭐ {row['Rating_pengguna']}")
                    st.markdown(f"**Kategori Rekomendasi:** 🎯 {row['Prediksi_Kelas']}")
                st.markdown("<hr>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Hasil Rekomendasi",
            data=hasil.to_csv(index=False),
            file_name="rekomendasi_hp.csv",
            mime="text/csv"
        )
else:
    st.info("Masukkan filter pada sidebar lalu klik *Cari Rekomendasi*.")
