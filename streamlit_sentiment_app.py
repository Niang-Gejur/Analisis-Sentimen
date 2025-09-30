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
        # Tidak perlu preprocessing lagi karena sudah clean
        X_new = vectorizer.transform(df_new["clean_text"].astype(str))
        preds = model.predict(X_new)
        df_new["predicted_sentiment"] = preds

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
