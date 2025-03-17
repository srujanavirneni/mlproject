import numpy as np
import pandas as pd 
import re
from time import time 
import spacy
from tqdm.notebook import tqdm

import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

# Try different encodings
encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']

# Removes non-alphabetic characters:
def text_cleaner(column):
    for row in column[:2]:  # Limit processing to first 2 rows
        row = re.sub(r"(\t|\r|\n)", ' ', str(row)).lower()  # Remove escape characters
        row = re.sub(r"(__+|--+|~~+|\+\+|\.\.+)", ' ', str(row)).lower()  # Remove repetitive symbols
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()  # Remove special characters
        row = re.sub(r"(mailto:|\\x9\d)", ' ', str(row)).lower()  # Remove mail-related content
        row = re.sub(r"([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower()  # Replace INC nums
        row = re.sub(r"([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower()  # Replace CM# and CHG#
        row = re.sub(r"(\.\s+|-+\s+|:\s+)", ' ', str(row)).lower()  # Remove trailing symbols
        row = re.sub(r"(\s+.\s+)", ' ', str(row)).lower()  # Remove single-character words
        row = re.sub(r"(\s+)", ' ', str(row)).lower().strip()  # Remove extra spaces
        
        # Replace any URL (e.g., https://abc.xyz.net/browse/sdf-5327 → abc.xyz.net)
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            if url:
                row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', url.group(3), str(row))
        except:
            pass
        
        yield row


def main():
    for encoding in encodings_to_try:
        try:
            # Read only the first 2 rows
            df1 = pd.read_csv('news_summary.csv', encoding=encoding, nrows=2)
            df2 = pd.read_csv('news_summary_more.csv', encoding=encoding, nrows=2)
            print(f"Successfully read CSV files using encoding: {encoding}")

            print(df1.columns)
            print(df2.columns)

            # Combine 'text' and 'headlines' columns from df1 and df2 (only 2 rows)
            syntext = pd.concat([df1['text'], df2['text']], ignore_index=True)
            summary = pd.concat([df1['headlines'], df2['headlines']], ignore_index=True)

            # Create a new DataFrame with 'text' and 'summary' columns
            data = pd.DataFrame({'text': syntext, 'summary': summary})

            print(data.head())

            clean_text = list(text_cleaner(data['text']))
            clean_summary = list(text_cleaner(data['summary']))

            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # Load Spacy model
            
            t = time()
            cleaned_text = [str(doc) for doc in nlp.pipe(clean_text, batch_size=2, n_process=1)]
            print(f'Time to clean text: {round((time() - t) / 60, 2)} mins')

            t = time()
            cleaned_summaries = ['_START ' + str(doc) + ' _END_' for doc in nlp.pipe(clean_summary, batch_size=2, n_process=1)]
            print(f'Time to clean summary: {round((time() - t) / 60, 2)} mins')

            data1 = pd.DataFrame({
                'clean_txt': cleaned_text,
                'clean_summary': cleaned_summaries
            })

            print(data1)

            break  # Stop trying encodings if successful
        except UnicodeDecodeError:
            print(f"Failed to read CSV files using encoding: {encoding}")

if __name__ == "__main__":
    main()
