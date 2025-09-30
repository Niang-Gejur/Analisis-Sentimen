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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

    return model, vectorizer

# ===============================
# 3. STREAMLIT APP
# ===============================
st.title("📊 Sentiment Analysis App (Naïve Bayes + Streamlit)")

# Train model once
model, vectorizer = train_model()

# ===============================
# 4. UPLOAD DATASET BARU
# ===============================
st.subheader("📂 Upload Dataset Baru untuk Prediksi")
uploaded_file = st.file_uploader("Upload file .xlsx", type=["xlsx"])

if uploaded_file:
    df_new = pd.read_excel(uploaded_file)

    if "clean_text" not in df_new.columns:
        st.error("Kolom 'clean_text' tidak ditemukan dalam dataset!")
    else:
        X_new = vectorizer.transform(df_new["clean_text"].astype(str))
        preds = model.predict(X_new)
        df_new["predicted_sentiment"] = preds

        st.subheader("🔹 Hasil Prediksi (contoh 20 baris)")
        st.dataframe(df_new[["clean_text", "predicted_sentiment"]].head(20))

        # ===============================
        # 5. Evaluasi jika ada label
        # ===============================
        if "klasifikasi" in df_new.columns:
            y_true = df_new["klasifikasi"].astype(str)
            y_pred = df_new["predicted_sentiment"]

            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)

            st.subheader("📌 Evaluasi Model pada Dataset Baru")
            st.write("Akurasi:", round(acc, 4))
            st.text(report)

            # Confusion Matrix pakai matplotlib (tanpa seaborn)
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")

            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_xticks(range(len(np.unique(y_true))))
            ax.set_yticks(range(len(np.unique(y_true))))
            ax.set_xticklabels(np.unique(y_true))
            ax.set_yticklabels(np.unique(y_true))
            st.pyplot(fig)

        # ===============================
        # 6. Distribusi Sentimen
        # ===============================
        st.subheader("📊 Distribusi Sentimen")
        st.bar_chart(df_new["predicted_sentiment"].value_counts())

        # ===============================
        # 7. Wordcloud per kelas
        # ===============================
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
