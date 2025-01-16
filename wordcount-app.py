import streamlit as st
import fitz  # PyMuPDF for extracting text from PDFs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    
    # Remove stopwords
    eng_stopwords = set(stopwords.words('english'))
    words = [word for word in words if word not in eng_stopwords]
    
    return words

# Streamlit app
st.title("PDF Text Preprocessing and Word Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Extract text
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    
    # Preprocess text
    with st.spinner("Processing text..."):
        words = preprocess_text(raw_text)
        word_count = Counter(words)
    
    # Display most common words
    st.subheader("Most Common Words")
    num_words = st.slider("Number of words to display:", 5, 50, 20)
    most_common_words = word_count.most_common(num_words)
    st.bar_chart(pd.DataFrame(most_common_words, columns=["Word", "Frequency"]).set_index("Word"))
    
    # Generate Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, colormap="viridis").generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)  # Display the word cloud image
    
    # Display scrolling table
    st.subheader("Word Frequency Table")
    word_freq_df = pd.DataFrame(word_count.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    st.dataframe(word_freq_df, height=400)

    st.success("Processing complete!")