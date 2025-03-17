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
    for row in column:
        
        # ORDER OF REGEX IS VERY IMPORTANT!!!!!!
        
        row = re.sub(r"(\t)", ' ', str(row)).lower()  # Remove escape characters
        row = re.sub(r"(\r)", ' ', str(row)).lower() 
        row = re.sub(r"(\n)", ' ', str(row)).lower()
        
        row = re.sub(r"(__+)", ' ', str(row)).lower()   # Remove consecutive underscores
        row = re.sub(r"(--+)", ' ', str(row)).lower()   # Remove consecutive hyphens
        row = re.sub(r"(~~+)", ' ', str(row)).lower()   # Remove consecutive tildes
        row = re.sub(r"(\+\++)", ' ', str(row)).lower()   # Remove consecutive plus signs
        row = re.sub(r"(\.\.+)", ' ', str(row)).lower()   # Remove consecutive dots
        
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()  # Remove special characters
        
        row = re.sub(r"(mailto:)", ' ', str(row)).lower()  # Remove mailto:
        row = re.sub(r"(\\x9\d)", ' ', str(row)).lower()  # Remove \x9* in text
        row = re.sub(r"([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower()  # Replace INC nums to INC_NUM
        row = re.sub(r"([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower()  # Replace CM# and CHG# to CM_NUM
        
        row = re.sub(r"(\.\s+)", ' ', str(row)).lower()  # Remove full stop at end of words
        row = re.sub(r"(-\s+)", ' ', str(row)).lower()  # Remove - at end of words
        row = re.sub(r"(:\s+)", ' ', str(row)).lower()  # Remove : at end of words
        
        row = re.sub(r"(\s+.\s+)", ' ', str(row)).lower()  # Remove single characters hanging between spaces

        # Replace any URL like https://abc.xyz.net/browse/sdf-5327 with abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(row))
        except:
            pass  # There might be emails with no URL in them
        
        row = re.sub(r"(\s+)", ' ', str(row)).lower()  # Remove multiple spaces
        
        # Should always be last
        row = re.sub(r"(\s+.\s+)", ' ', str(row)).lower()  # Remove any single characters hanging between spaces
        
        yield row


def main():
    for encoding in encodings_to_try:
        try:
            df1 = pd.read_csv('news_summary.csv', encoding=encoding)
            df2 = pd.read_csv('news_summary_more.csv', encoding=encoding)
            print(f"Successfully read CSV files using encoding: {encoding}")

            print(df1.columns)
            print(df2.columns)

            # Combine 'text' columns from df1 and df2
            syntext = pd.concat([df1['text'], df2['text']], ignore_index=True)

            # Combine 'headlines' columns from df1 and df2
            summary = pd.concat([df1['headlines'], df2['headlines']], ignore_index=True)

            # Create a new DataFrame with 'syntext' and 'Summary' columns
            data = pd.DataFrame({'text': syntext, 'summary': summary})

            print(data.head(1))

            clean_text = text_cleaner(data['text'])
            clean_summary = text_cleaner(data['summary'])

            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # Load English model
            t = time()
            cleaned_text = [str(doc) for doc in tqdm(nlp.pipe(clean_text, batch_size=5000, n_process=-1))]
            print('Time to Clean up Data: {} Mins'.format(round((time() - t) / 60, 2)))

            t = time()
            cleaned_summaries = ['_START ' + str(doc) + ' _END_' for doc in tqdm(nlp.pipe(clean_summary, batch_size=5000, n_process=-1))]
            print('Time to clean up summary data: {} Mins'.format(round((time() - t)/ 60, 2)))

            data1 = pd.DataFrame({
                'clean_txt': cleaned_text,
                'clean_summary': cleaned_summaries
            })

            print(cleaned_text[1])
            print(cleaned_summaries[1])

            break  # Stop trying encodings if successful
        except UnicodeDecodeError:
            print(f"Failed to read CSV files using encoding: {encoding}")

if __name__ == "__main__":
    main()
