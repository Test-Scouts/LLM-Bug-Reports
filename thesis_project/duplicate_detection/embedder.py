import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


# this function should be called when the program is started to produce the embeddings file
def embed_bug_reports():
    print('embed bug reporttayiz')
    # pandas DataFrame containing bug report data
    df = pd.read_csv('duplicate_detection/data/testscouts/bug_reports.csv', sep=';', on_bad_lines='skip', usecols=["Description",
                                                                                                        "Issue id"])
    df = df.replace('\n', '', regex=True)

    # to store bug report descriptions as strings
    descriptions = []

    for row in range(df.shape[0]):
        # stringify description values and append to sentences
        sentences = str(df.values[row][1])
        descriptions.append(sentences)

    #model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    #mert's change over here
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    #mert's change ends here

    bug_report_embeddings = model.encode(descriptions)

    # Store sentences & embeddings on disc
    with open('embeddings.pkl', "wb") as fOut:
        pickle.dump({'embeddings': bug_report_embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)


# this function should be called for loading the embeddings file when finding duplicates
def load_embeddings():
    # Load sentences & embeddings from disc
    return np.load('duplicate_detection/embeddings/testscouts/embeddings_bigLLM_1548.npy')

