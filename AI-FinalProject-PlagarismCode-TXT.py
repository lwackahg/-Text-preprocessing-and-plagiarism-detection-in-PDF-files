"""
This script detects plagiarism in a given PDF file by comparing its content with other PDF files in a specified directory.
It uses the NLTK library for text preprocessing, the TfidfVectorizer from scikit-learn for text representation, and cosine
similarity as a measure of similarity between documents.
"""

import difflib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import os
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    tokens = word_tokenize(text.lower()) # Tokenizing the text
    stop_words = set(stopwords.words('english')) # Loading the stop words
    tokens = [token for token in tokens if token not in stop_words] # Removing stop words from the tokens
    lemmatizer = WordNetLemmatizer() # Initializing the lemmatizer
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # Lemmatizing the tokens
    return ' '.join(tokens)

with open('/content/test/suspicious 10.pdf', 'rb') as f: # Set Suspicious File
    pdf_reader = PyPDF2.PdfReader(f) # Creating a PDF reader object
    text2 = ''
    for page in pdf_reader.pages: # Iterating through the pages in the PDF
        text2 += page.extract_text() # Extracting text from each page
text2 = preprocess(text2) # Preprocessing the extracted text
texts = [text2]
file_names = ['/content/test/suspicious 10.pdf']

pdf_dir = "/content"
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

for pdf_file in pdf_files: # All Other Files to be Read
    with open(pdf_file, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text1 = ''
        for page in pdf_reader.pages:
            text1 += page.extract_text()

    text1 = preprocess(text1)  # Preprocessing the extracted text
    texts.append(text1)  # Appending the preprocessed text to the list of texts
    file_names.append(pdf_file)  # Appending the PDF file name to the list of file names


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

for idx, score in enumerate(similarity_scores[0]):
    if score > 0.5: # If the similarity score is greater than 0.5, report plagiarism
        print(f"Plagiarism detected in {file_names[idx + 1]} with similarity score: {score}")
    else: # If the similarity score is less than or equal to 0.5, report no plagiarism
        print(f"No plagiarism detected in {file_names[idx + 1]}")
