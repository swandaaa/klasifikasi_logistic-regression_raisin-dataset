import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Muat model KNN yang telah dilatih
loaded_scaler = joblib.load('scaler.pkl')

# Muat model KNN yang telah dilatih
LR_model = joblib.load('LR_model.pkl')

# Fungsi prediksi menggunakan model KNN dan PCA
def raisin_classification(area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter):
    area = float(area)
    MajorAxisLength = float(MajorAxisLength)
    MinorAxisLength = float(MinorAxisLength)
    Eccentricity = float(Eccentricity)
    ConvexArea = float(ConvexArea)
    Extent = float(Extent)
    Perimeter = float(Perimeter)

    # Kumpulkan semua fitur dalam satu DataFrame
    features_df = pd.DataFrame({
        'Area': [area],
        'MajorAxisLength': [MajorAxisLength],
        'MinorAxisLength': [MinorAxisLength],
        'Eccentricity': [Eccentricity],
        'ConvexArea': [ConvexArea],
        'Extent': [Extent],
        'Perimeter': [Perimeter]
    })

    # Menampilkan tabel fitur sebelum di normalisasi
    st.write("Hasil Ekstraksi Audio Sebelum Normalisasi")
    st.dataframe(features_df)

    # Menggunakan scaler untuk transformasi data
    normalized_data = loaded_scaler.transform(features_df)

    # Menampilkan tabel fitur setelah di normalisasi
    st.write("Hasil Ekstraksi Audio Setelah Normalisasi")
    st.dataframe(pd.DataFrame(normalized_data, columns=features_df.columns))

    # Prediksi emosi menggunakan model KNN
    prediction = LR_model.predict(normalized_data)
    
    return prediction[0]

def main():
    st.title('Aplikasi Klasifikasi Jenis Kismis')

    # Input parameter area
    area_input = st.text_input("Masukkan nilai area:")
    # Input parameter MajorAxisLength
    MajorAxisLength_input = st.text_input("Masukkan nilai MajorAxisLength:")
    # Input parameter MinorAxisLength
    MinorAxisLength_input = st.text_input("Masukkan nilai MinorAxisLength:")
    # Input parameter Eccentricity
    Eccentricity_input = st.text_input("Masukkan nilai Eccentricity:")
    # Input parameter ConvexArea
    ConvexArea_input = st.text_input("Masukkan nilai ConvexArea:")
    # Input parameter Extent
    Extent_input = st.text_input("Masukkan nilai Extent:")
    # Input parameter Perimeter
    Perimeter_input = st.text_input("Masukkan nilai Perimeter:")
    
    # Melakukan prediksi
    if st.button('Prediksi'):
        # Memanggil fungsi prediksi
        prediction = raisin_classification(area_input, MajorAxisLength_input, MinorAxisLength_input, Eccentricity_input, ConvexArea_input, Extent_input, Perimeter_input)
        st.write('Hasil Prediksi:')
        st.write(prediction)

if __name__ == "__main__":
    main()
