import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
# 2. TRAINING MODEL
# ===============================
@st.cache_resource
def train_model():
    df_train = pd.read_excel("labeled_tweets.xlsx", sheet_name="Data Hasil")
    text_col = "stemmed_text"
    label_col = "klasifikasi"
    df_train = df_train.dropna(subset=[text_col, label_col]).copy()

    X = df_train[text_col].astype(str)
    y = df_train[label_col].astype(str)

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    y_pred = model.predict(X_vec)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=False)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)

    return model, vectorizer, acc, report, cm, model.classes_

# ===============================
# 3. STREAMLIT APP
# ===============================
st.title("📊 Sentiment Analysis App (Naïve Bayes + Streamlit)")

# Train model once
model, vectorizer, acc, report, cm, labels = train_model()

# ===============================
# 4. UPLOAD DATASET BARU
# ===============================
st.subheader("📂 Upload Dataset Baru untuk Prediksi")
uploaded_file = st.file_uploader("Upload file .xlsx", type=["xlsx"])

if uploaded_file:
    df_new = pd.read_excel(uploaded_file)

    # cek kolom teks
    if "clean_text" in df_new.columns:
        st.info("✅ Kolom 'clean_text' sudah tersedia, dipakai langsung.")
    elif "full_text" in df_new.columns:
        st.info("ℹ️ Membuat kolom 'clean_text' dari 'full_text'.")
        df_new["clean_text"] = df_new["full_text"].astype(str).apply(preprocess_text)
    else:
        st.error("Dataset harus punya kolom 'clean_text' atau 'full_text'!")
        st.stop()

    # transformasi & prediksi
    X_new = vectorizer.transform(df_new["clean_text"])
    preds = model.predict(X_new)
    df_new["predicted_sentiment"] = preds

    # ===============================
    # 5. EVALUASI MODEL (DATA TRAINING)
    # ===============================
    st.subheader("🔹 Evaluasi Model (Data Training)")
    st.write("Akurasi Training:", round(acc, 4))
    st.text(report)

    # Confusion Matrix Training
    st.subheader("📉 Confusion Matrix (Data Training)")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # ===============================
    # 6. HASIL PREDIKSI DATASET BARU
    # ===============================
    st.subheader("🔹 Hasil Prediksi (contoh 20 baris)")
    st.dataframe(df_new[["clean_text", "predicted_sentiment"]].head(20))

    # Distribusi Sentimen
    st.subheader("📊 Distribusi Sentimen")
    st.bar_chart(df_new["predicted_sentiment"].value_counts())

    # Wordcloud per kelas
    st.subheader("☁️ Wordcloud per Sentimen")
    sentiments = df_new["predicted_sentiment"].unique()
    for sent in sentiments:
        text_data = " ".join(df_new[df_new["predicted_sentiment"] == sent]["clean_text"])
        if text_data.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(text_data)
            st.write(f"**Sentimen: {sent}**")
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # ===============================
    # 7. EVALUASI DATASET BARU (JIKA ADA LABEL)
    # ===============================
    if "klasifikasi" in df_new.columns:
        st.subheader("🔹 Evaluasi Dataset Baru (dengan Label Asli)")
        y_true = df_new["klasifikasi"].astype(str)
        y_pred = df_new["predicted_sentiment"]

        acc_new = accuracy_score(y_true, y_pred)
        report_new = classification_report(y_true, y_pred, output_dict=False)
        cm_new = confusion_matrix(y_true, y_pred, labels=model.classes_)

        st.write("Akurasi Dataset Baru:", round(acc_new, 4))
        st.text(report_new)

        # Confusion Matrix Baru
        st.subheader("📉 Confusion Matrix (Dataset Baru)")
        fig, ax = plt.subplots()
        disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=labels)
        disp_new.plot(ax=ax, cmap="Oranges", colorbar=False)
        st.pyplot(fig)
