import numpy as np
from helper import remove_punc
from hw8_1 import *
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords


# Clean and prepare the contents of a document
def read_and_clean_doc(doc):
    # 1. Open document, read text into *single* string
    with open(doc, "r") as f:
        allStr = f.read()

    # 2. Filter out punctuation from list of words (use remove_punc)
    all_rm_punc = remove_punc(allStr)

    # 3. Make the words lower case
    all_lower = all_rm_punc.lower()

    # 4. Filter out stopwords
    tok_low = all_lower.split(" ")
    filt_tok = [tok for tok in tok_low if tok not in stopwords.words("english")]
    all_no_stop = "".join(filt_tok)

    return all_no_stop


# Builds a doc-word matrix
def build_doc_word_matrix(doclist, n, normalize=False):
    doc_dicts = []
    all_ngrams = set()

    for doc in doclist:
        cleaned = read_and_clean_doc(doc)
        ngrams = get_ngrams(cleaned, n)

        counts = {}
        for gram in ngrams:
            counts[gram] = counts.get(gram, 0) + 1

        doc_dicts.append(counts)
        all_ngrams.update(counts.keys())

    ngramlist = sorted(all_ngrams)
    docword = np.zeros((len(doclist), len(ngramlist)))

    for i, counts in enumerate(doc_dicts):
        for j, gram in enumerate(ngramlist):
            docword[i, j] = counts.get(gram, 0)
            
    if normalize:
        totals = docword.sum(axis=1)
        docword = np.divide(docword, totals[:, np.newaxis], out=np.zeros_like(docword), where=totals[:, np.newaxis] != 0)
        docword = np.round(docword, 4)
    return docword, ngramlist


# Builds a term-frequency matrix
def build_tf_matrix(docword):
    return  np.divide(docword, docword.sum(axis=1)[:, np.newaxis], out=np.zeros_like(docword, dtype=float), where=docword.sum(axis=1)[:, np.newaxis] != 0)


# Builds an inverse document frequency matrix
def build_idf_matrix(docword):
    num_docs = docword.shape[0]
    doc_freq = np.count_nonzero(docword, axis=0)

    idf = np.zeros(docword.shape[1])
    np.divide(num_docs, doc_freq, out=idf, where=doc_freq != 0)

    return np.log10(idf, out=np.zeros_like(idf), where=idf > 0)


#   Builds a tf-idf matrix given a doc word matrix
def build_tfidf_matrix(docword):
    return build_tf_matrix(docword) * build_idf_matrix(docword)


#   Find the three most distinctive ngrams, according to TFIDF, in each document
def find_distinctive_ngrams(docword, ngramlist, doclist):
    distinctive_words = {}
    tfidf = build_tfidf_matrix(docword)

    for i, doc in enumerate(doclist):
        indices = np.argsort(-tfidf[i], kind="stable")[:3]
        distinctive_words[doc] = [ngramlist[j] for j in indices]
    return distinctive_words


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join, splitext

    ### Test Cases ###
    directory = "lecs"
    path1 = join(directory, "1_vidText.txt")
    path2 = join(directory, "2_vidText.txt")

    print("\n*** Testing build_doc_word_matrix ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 4)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    print("\n*** Testing build_doc_word_matrix normalization ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 4, normalize=True)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    
    print("\n*** Testing build_tf_matrix ***")
    tf = build_tf_matrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis=1))
    print("\n*** Testing build_idf_matrix ***")
    idf = build_idf_matrix(docword)
    print(idf[0][0:10])
    print("\n*** Testing build_tfidf_matrix ***")
    tfidf = build_tfidf_matrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])

    print("\n*** Testing find_distinctive_words ***")
    print(find_distinctive_ngrams(docword, wordlist, doclist))
