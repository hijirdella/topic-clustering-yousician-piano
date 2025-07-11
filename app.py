import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, time
import pytz

# === Load Model dan Vectorizer ===
kmeans_model = joblib.load("Yousician_Piano_clustering.pkl")
tfidf_vectorizer = joblib.load("Yousician_Piano_tfidf_vectorizer.pkl")

# === Set Halaman ===
st.set_page_config(page_title="Topic Clustering - Yousician: Learn Piano", layout="wide")

# === Judul Aplikasi dengan Emoji Piano ===
st.title("🎹 Topic Clustering - Yousician: Learn Piano")

# === Fungsi Pembersihan Review ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Fungsi Prediksi Cluster ===
def predict_cluster(texts):
    X = tfidf_vectorizer.transform(texts)
    clusters = kmeans_model.predict(X)

    if X.shape[0] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())
    else:
        X_pca = [[0.0, 0.0]]

    return clusters, X_pca

# === Pilihan Mode Input ===
mode = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload CSV"])

# === MODE 1: INPUT MANUAL ===
if mode == "📝 Input Manual":
    name = st.text_input("Nama Pengguna:")
    star_rating = st.selectbox("Rating Bintang:", [1, 2, 3, 4, 5])

    wib = pytz.timezone("Asia/Jakarta")
    now_wib = datetime.now(wib)

    review_day = st.date_input("📅 Tanggal Ulasan:", value=now_wib.date())
    review_time = st.time_input("⏰ Waktu Ulasan:", value=now_wib.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_datetime_wib = wib.localize(review_datetime)
    review_date_str = review_datetime_wib.strftime("%Y-%m-%d %H:%M")

    review = st.text_area("Tulis Review di sini:")

    if st.button("Prediksi Cluster"):
        if review.strip() == "":
            st.warning("Review tidak boleh kosong.")
        else:
            cleaned = clean_text(review)
            cluster, pca_result = predict_cluster([cleaned])
            df_result = pd.DataFrame({
                "Name": [name],
                "Star Rating": [star_rating],
                "Datetime (WIB)": [review_date_str],
                "Review": [review],
                "Cluster": cluster,
                "PCA 1": pca_result[0][0],
                "PCA 2": pca_result[0][1]
            })
            st.dataframe(df_result)

            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_result, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=100, ax=ax)
            st.pyplot(fig)

# === MODE 2: UPLOAD CSV ===
else:
    file = st.file_uploader("Upload file CSV dengan kolom: name, star_rating, date, review", type="csv")
    if file:
        df = pd.read_csv(file)

        required_cols = {'name', 'star_rating', 'date', 'review'}
        if not required_cols.issubset(df.columns):
            st.error(f"File harus memiliki kolom: {', '.join(required_cols)}")
        else:
            df['cleaned_review'] = df['review'].fillna("").apply(clean_text)
            clusters, pca_result = predict_cluster(df['cleaned_review'])
            df['Cluster'] = clusters
            df['PCA 1'] = [x[0] for x in pca_result]
            df['PCA 2'] = [x[1] for x in pca_result]

            st.dataframe(df[['name', 'star_rating', 'date', 'review', 'Cluster']])

            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=70, ax=ax)
            st.pyplot(fig)

            st.download_button(
                label="📥 Unduh Hasil Klaster",
                data=df.to_csv(index=False),
                file_name="hasil_klaster_yousician_learn_piano.csv",
                mime="text/csv"
            )
