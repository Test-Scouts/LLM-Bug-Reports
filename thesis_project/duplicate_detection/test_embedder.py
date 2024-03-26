import h5py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


def embed_bug_reports():
    # pandas DataFrame containing bug report data
    df = pd.read_csv('data/mozilla_firefox.csv', sep=',', on_bad_lines='skip', usecols=["Issue_id", "Description"])
    df = df.replace('\n', '', regex=True)
    print(df.values)
    # to store bug report descriptions as strings
    descriptions = []

    for row in range(df.shape[0]):
        # stringify description values and append to sentences
        sentences = str(df.values[row][1])
        descriptions.append(sentences)

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    bug_report_embeddings = model.encode(descriptions)

    # Store embeddings on disc in NPY format
    np.save('test_embeddings.npy', bug_report_embeddings)


# this function should be called for loading the embeddings file when finding duplicates
def load_embeddings():
    # Load embeddings from disc in NPY format
    stored_embeddings = np.load('test_embeddings.npy')
    return stored_embeddings


embed_bug_reports()
