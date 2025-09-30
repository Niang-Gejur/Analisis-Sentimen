# streamlit_sentiment_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from textblob import TextBlob

nltk.download("punkt")
nltk.download("wordnet")

# ===============================
# 1. PREPROCESSING
# ===============================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hanya huruf
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# 2. AUTO LABELING DENGAN TEXTBLOB
# ===============================
def auto_label_with_textblob(df, text_col="clean_text"):
    status = []
    total_positif = total_negatif = total_netral = 0

    for tweet in df[text_col]:
        analysis = TextBlob(str(tweet))
        if analysis.sentiment.polarity > 0.0:
            status.append("Positif")
            total_positif += 1
        elif analysis.sentiment.polarity == 0.0:
            status.append("Netral")
            total_netral += 1
        else:
            status.append("Negatif")
            total_negatif += 1

    df["klasifikasi"] = status
    return df, total_positif, total_netral, total_negatif

# ===============================
# 3. STREAMLIT APP
# ===============================
st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES")

vectorizer = TfidfVectorizer()
model = MultinomialNB()

# ===============================
# 4. UPLOAD DATASET BARU
# ===============================
st.subheader("📂 Upload Dataset Baru untuk Analisis")
uploaded_file = st.file_uploader("Upload file .xlsx", type=["xlsx"])

if uploaded_file:
    df_new = pd.read_excel(uploaded_file)

    # Cari kolom teks utama (prioritas: full_text > stemmed_text > clean_text)
    text_col = None
    for candidate in ["full_text", "stemmed_text", "clean_text"]:
        if candidate in df_new.columns:
            text_col = candidate
            break

    if text_col is None:
        st.error("Dataset harus punya salah satu kolom teks: 'full_text', 'stemmed_text', atau 'clean_text'.")
        st.stop()

    # Buat kolom clean_text
    if text_col != "clean_text":
        df_new["clean_text"] = df_new[text_col].astype(str).apply(preprocess_text)
    else:
        df_new["clean_text"] = df_new["clean_text"].astype(str)

    # Auto labeling jika kolom klasifikasi tidak ada
    if "klasifikasi" not in df_new.columns:
        st.info("Kolom 'klasifikasi' tidak ditemukan → Label otomatis dibuat dengan TextBlob.")
        df_new, pos, net, neg = auto_label_with_textblob(df_new, text_col="clean_text")
        st.success(f"Label otomatis selesai: Positif={pos}, Netral={net}, Negatif={neg}")

    # ===============================
    # 5. Split Data & Training
    # ===============================
    X = df_new["clean_text"].astype(str)
    y = df_new["klasifikasi"].astype(str)

    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.subheader("📊 Evaluasi Model (Train/Test Split)")
    st.write("Akurasi:", round(acc, 4))
    st.text(report)

    # ===============================
    # 6. Confusion Matrix
    # ===============================
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Versi Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Heatmap")
    st.pyplot(fig)

    # ===============================
    # 7. Hasil Prediksi
    # ===============================
    st.subheader("🔹 Hasil Prediksi (contoh 20 baris)")
    df_new["predicted_sentiment"] = model.predict(X_vec)
    st.dataframe(df_new[[text_col, "predicted_sentiment"]].head(20))

    # ===============================
    # 9. Wordcloud per Sentimen
    # ===============================
    st.subheader("☁️ Wordcloud per Sentimen")
    sentiments = sentiment_counts.index  # gunakan urutan lengkap
    colors = {"Positif": "Greens", "Netral": "Greys", "Negatif": "Reds"}

    for sent in sentiments:
        text_data = " ".join(df_new[df_new["predicted_sentiment"] == sent]["clean_text"])
        if text_data.strip():
            wc = WordCloud(width=600, height=400, background_color="white", colormap=colors[sent]).generate(text_data)
            st.write(f"**Sentimen: {sent}**")
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # ===============================
    # 10. Download Hasil Prediksi
    # ===============================
    csv = df_new.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Hasil Prediksi (CSV)",
        data=csv,
        file_name="hasil_sentimen.csv",
        mime="text/csv",
    )
