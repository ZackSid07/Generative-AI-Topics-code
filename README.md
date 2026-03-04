# NLP Data Preprocessing and Cleaning Pipeline

## Overview
This repository provides a foundational toolkit for cleaning and preparing raw text data for Natural Language Processing (NLP) tasks. Text data in its raw form (especially from web scraping or social media) is often noisy, unstructured, and filled with artifacts that can degrade the performance of machine learning models. 

This project contains modular, easy-to-understand Python functions designed to standardize text, remove irrelevant noise, and reduce the dimensionality of your dataset before feeding it into a model.

## Features and Methodologies
The current pipeline includes the following core preprocessing steps:

* **Lowercasing:** Converts all text to a uniform lower case to prevent the model from treating identical words with different casings as unique entities.
* **HTML Tag Removal:** Strips out web formatting tags (e.g., `<br />`, `<p>`) using regular expressions, which is essential for web-scraped datasets.
* **URL Removal:** Eliminates hyperlinks (`http`, `https`, `www`), as they generally do not contribute contextual or sentimental value to NLP models.
* **Punctuation Removal:** Clears grammatical markers using Python's optimized `string.translate` method, preventing punctuation from attaching to and altering words.
* **Chat Word Conversion:** Translates common internet slang and abbreviations (e.g., "ASAP", "IMO") into standard English using a predefined dictionary lookup.
* **Spelling Correction:** Utilizes the `TextBlob` library to automatically detect and correct spelling errors, ensuring misspellings are not treated as out-of-vocabulary words.
* **Stopword Removal:** Filters out highly frequent but low-meaning words (e.g., "the", "is", "in") using the `nltk` library to reduce dataset size and computation time.
* **Emoji Handling:** Converts emojis into their textual representations (e.g., `: fire: ') using the `emoji` library, preserving the emotional sentiment of the text.
* **Tokenization:** Breaks down continuous text strings into individual tokens (words or sentences) using `nltk` and `spaCy`, setting the stage for vectorization.
* **Stemming and Lemmatization:** Reduces words to their base or dictionary form. This repository emphasizes Lemmatization (via `WordNetLemmatizer`) for its grammatically accurate root extraction, while also providing Stemming (via `PorterStemmer`) for faster, less rigid applications.
## Dependencies
To utilize the scripts in this repository, ensure you have the following Python libraries installed:
```bash
pip install pandas textblob nltk emoji spacy
python -m spacy download en_core_web_sm

You will also need to download the required NLTK corpora. Run the following within your Python environment:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

Usage
The functions provided in this repository are highly modular. You can apply them individually to a Pandas DataFrame column using the .apply() method, or chain them together to create a custom preprocessing pipeline tailored to your specific dataset's needs.

# Example of applying a single function to a pandas DataFrame
import pandas as pd
from preprocessing_module import remove_html_tags, remove_stopwords

df = pd.read_csv('your_dataset.csv')
df['cleaned_text'] = df['raw_text'].apply(remove_html_tags).apply(remove_stopwords)



```bash
pip install pandas textblob nltk emoji spacy
python -m spacy download en_core_web_sm
