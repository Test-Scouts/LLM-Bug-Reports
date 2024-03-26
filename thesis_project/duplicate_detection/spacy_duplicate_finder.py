import pandas as pd
import spacy_universal_sentence_encoder


def get_similar_bugs(description):
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    df = pd.read_csv('duplicate_detection/data/bug_reports.csv', sep=';', on_bad_lines='skip', usecols=["Description"])
    df = df.replace('\n', '', regex=True)
    list_result = []
    doc_2 = nlp(description)
    result = float(0)

    for row in range(df.shape[0]):
        doc_1 = nlp(str(df.values[row]))
        a = doc_2.similarity(doc_1)
        if a > result:
            temp = []
            result = float(a)
            r = round(result, 3)
            temp.append(r)
            temp.append(str(doc_1))
            list_result.append(temp)

    match = list_result[-3:]
    print(match)
    return match
