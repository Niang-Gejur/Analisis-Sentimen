# streamlit_sentiment_app.py
# Streamlit app for sentiment analysis (adapted to user's datasets)
# Inspired by: https://supertype.ai/notes/twitter-sentiment-analysis-part-3
# Author: ChatGPT (adapted for provided datasets)

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Make sure required NLTK data is available (first-run will download)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# ----------------------------- Helper functions -----------------------------

def clean_text(text):
    """Basic cleaning + lemmatization adapted from the reference pipeline."""
    try:
        text = str(text).lower()
        # replace URLs, mentions, entities
        text = re.sub(r"((http|https)://[^\s]+)|www\.[^\s]+", ' ', text)
        text = re.sub(r"@[^\s]+", ' ', text)
        text = re.sub(r"&\w+;", ' ', text)
        # replace non alpha characters with space
        text = re.sub(r"[^a-z ]", ' ', text)
        # handle n't -> not
        text = re.sub(r"n't\b", ' not', text)
        tokens = word_tokenize(text)
        word_tag_tuples = pos_tag(tokens, tagset='universal')
        tag_dict = {'NOUN':'n', 'VERB':'v', 'ADJ':'a', 'ADV':'r'}
        lemmatizer = WordNetLemmatizer()
        final_tokens = []
        for w, tag in word_tag_tuples:
            if len(w) > 1:
                if tag in tag_dict:
                    final_tokens.append(lemmatizer.lemmatize(w, tag_dict[tag]))
                else:
                    final_tokens.append(lemmatizer.lemmatize(w))
        return ' '.join(final_tokens)
    except Exception:
        return ''


def load_example_datasets():
    """Try to load common filenames the user uploaded. If not present, return None."""
    candidates = ['labeled_tweets.xlsx', 'mobilelegends_gabungan.xlsx', 'data.csv']
    for f in candidates:
        try:
            df = pd.read_excel(f) if f.endswith('.xlsx') else pd.read_csv(f)
            return df
        except Exception:
            continue
    return None


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label', ylabel='True label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig


def create_wordcloud(series, stopwords=None, max_words=100):
    text = ' '.join(series.dropna().astype(str))
    wc = WordCloud(background_color='white', stopwords=stopwords, max_words=max_words, collocations=False)
    wc.generate(text)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig

# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title='Sentiment Analyzer (Streamlit)', layout='wide')
st.title('📊 Sentiment Analysis Deployment (Streamlit)')
st.markdown('A Streamlit app adapted from the Supertype deployment guide. You can upload your dataset or use an example file (e.g., `mobilelegends_gabungan.xlsx`).')

# Sidebar: dataset input
st.sidebar.header('Data / Model')
uploaded_file = st.sidebar.file_uploader('Upload dataset (CSV or XLSX)', type=['csv','xlsx'])
use_example = st.sidebar.checkbox('Try to auto-load example dataset from workspace (labeled_tweets.xlsx or mobilelegends_gabungan.xlsx)')

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success(f'Loaded {uploaded_file.name} — shape: {df.shape}')
    except Exception as e:
        st.error('Failed to read uploaded file: ' + str(e))
        st.stop()
elif use_example:
    df = load_example_datasets()
    if df is None:
        st.warning('No example dataset found in workspace. Please upload one.')
        st.stop()
    else:
        st.success(f'Loaded example dataset — shape: {df.shape}')
else:
    st.info('Upload a dataset or check "Try to auto-load example dataset" to proceed.')
    st.stop()

# Show columns and allow user to select text & label columns
st.subheader('Dataset preview and column selection')
with st.expander('Preview data (first 10 rows)'):
    st.dataframe(df.head(10))

cols = df.columns.tolist()
text_col = st.selectbox('Select text column', options=cols, index=0)
label_col = st.selectbox('Select label column (if present). If none, you can leave blank to do unsupervised viz)', options=[None]+cols, index=cols.index(cols[1]) if len(cols)>1 else 0)

# If label column selected, show label distribution
if label_col:
    st.write('Label distribution:')
    st.bar_chart(df[label_col].value_counts())

# Preprocessing
st.subheader('Preprocessing')
run_prep = st.button('Run preprocessing')
if run_prep:
    with st.spinner('Cleaning text...'):
        df['clean_text'] = df[text_col].astype(str).apply(clean_text)
        st.success('Preprocessing finished. Sample:')
        st.dataframe(df[[text_col,'clean_text']].head(10))

# Training
st.subheader('Train / Evaluate model')
vectorizer_choice = st.selectbox('Vectorizer', options=['CountVectorizer','TfidfVectorizer'])
ngram_range = st.selectbox('N-gram range', options=['1,1','1,2','1,3'])
ngram_tuple = tuple(int(x) for x in ngram_range.split(','))
model_btn = st.button('Train model (MultinomialNB)')

if model_btn:
    if label_col is None:
        st.error('You must select a label column to train supervised model.')
    else:
        # drop missing labels or empty cleaned text
        df_train = df[[text_col, label_col, 'clean_text']].dropna(subset=[label_col]).copy()
        df_train['clean_text'] = df_train['clean_text'].replace('', np.nan)
        df_train = df_train.dropna(subset=['clean_text'])
        X = df_train['clean_text']
        y = df_train[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if vectorizer_choice == 'CountVectorizer':
            vect = CountVectorizer(ngram_range=ngram_tuple)
        else:
            vect = TfidfVectorizer(ngram_range=ngram_tuple)
        X_tr = vect.fit_transform(X_train)
        X_te = vect.transform(X_test)

        clf = MultinomialNB()
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        st.success(f'Training finished — test accuracy: {acc:.4f}')
        st.write('Classification report:')
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        fig_cm = plot_confusion(cm, labels=np.unique(y))
        st.pyplot(fig_cm)

        # Save model + vectorizer to allow quick prediction in app
        save_btn = st.button('Save model & vectorizer to files')
        if save_btn:
            joblib.dump({'vectorizer':vect, 'model':clf}, 'sentiment_pipeline.joblib')
            st.success('Saved to sentiment_pipeline.joblib in working directory.')

# Visualization: wordclouds & n-grams (if clean_text exists)
st.subheader('Visualizations')
if 'clean_text' not in df.columns:
    st.info('Run preprocessing to generate `clean_text` column for visualizations.')
else:
    viz_choice = st.selectbox('Choose visualization', options=['Wordcloud (all)','Wordcloud by label','Top unigrams','Top bigrams'])
    if viz_choice == 'Wordcloud (all)':
        fig_wc = create_wordcloud(df['clean_text'])
        st.pyplot(fig_wc)
    elif viz_choice == 'Wordcloud by label':
        if label_col is None:
            st.warning('Label column required for per-label wordclouds.')
        else:
            labels = df[label_col].dropna().unique()
            cols = st.columns(len(labels))
            for i, lab in enumerate(labels):
                with cols[i]:
                    st.write(f'Wordcloud for {lab}')
                    fig = create_wordcloud(df.loc[df[label_col]==lab,'clean_text'])
                    st.pyplot(fig)
    elif viz_choice == 'Top unigrams' or viz_choice == 'Top bigrams':
        n = st.slider('Top k', min_value=5, max_value=50, value=10)
        rng = (1,1) if viz_choice=='Top unigrams' else (2,2)
        vect = CountVectorizer(ngram_range=rng)
        X = vect.fit_transform(df['clean_text'].astype(str).values)
        words = vect.get_feature_names_out()
        counts = np.ravel(X.sum(axis=0))
        top_idx = counts.argsort()[::-1][:n]
        top_words = [(words[i], counts[i]) for i in top_idx]
        top_df = pd.DataFrame(top_words, columns=['term','count'])
        st.table(top_df)

# Inference section: load saved pipeline or use train on-the-fly
st.subheader('Predict single text (interactive)')
input_text = st.text_area('Enter text to predict', height=120)
use_saved = st.checkbox('Use saved pipeline file (sentiment_pipeline.joblib) if available')

if st.button('Predict'):
    if input_text.strip() == '':
        st.warning('Enter text to predict.')
    else:
        cleaned = clean_text(input_text)
        pipeline_loaded = None
        if use_saved:
            try:
                pipeline_loaded = joblib.load('sentiment_pipeline.joblib')
            except Exception:
                st.warning('No saved pipeline found; will try to use latest trained model in session if available.')
        if pipeline_loaded is not None:
            vect = pipeline_loaded['vectorizer']
            model = pipeline_loaded['model']
            X_in = vect.transform([cleaned])
            pred = model.predict(X_in)[0]
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_in).max()
            st.write('**Prediction**: ', pred)
            if proba is not None:
                st.write('Confidence: {:.3f}'.format(proba))
        else:
            st.info('No saved pipeline loaded. Attempting to train a quick lightweight model on the dataset (if label column available).')
            if label_col is None:
                st.error('Need a label column or a saved pipeline to predict.')
            else:
                # train quick pipeline
                df_tmp = df[[ 'clean_text', label_col]].dropna()
                df_tmp = df_tmp[df_tmp['clean_text']!='']
                if df_tmp.shape[0] < 10:
                    st.error('Not enough labeled rows to train a quick model.')
                else:
                    vect = CountVectorizer(ngram_range=(1,2))
                    X = vect.fit_transform(df_tmp['clean_text'])
                    y = df_tmp[label_col]
                    clf = MultinomialNB()
                    clf.fit(X, y)
                    X_in = vect.transform([cleaned])
                    pred = clf.predict(X_in)[0]
                    proba = clf.predict_proba(X_in).max() if hasattr(clf, 'predict_proba') else None
                    st.write('**Prediction**: ', pred)
                    if proba:
                        st.write('Confidence: {:.3f}'.format(proba))

st.markdown('---')
st.write('Notes: This app is an adaptation of the Streamlit deployment walkthrough. It allows dataset upload, preprocessing, quick model training (MultinomialNB), visualization (wordcloud, n-grams), and interactive prediction. For production/deep models (LSTM), follow the original guide which demonstrates loading tokenizers and Keras models.')

# End of file
