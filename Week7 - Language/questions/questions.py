import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1
filetosetofwords = {}

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    dict = {}

    files = [f for f in os.listdir(directory)]

    for file in files:
        with open(directory + os.sep + file) as f: 
            dict[file] = f.read()

    return dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    tokens = nltk.word_tokenize(document.lower())
    words = [token for token in tokens if token.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words("english"))    
    cleanData = [word for word in words if word not in stopwords]

    return cleanData


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    pool = []
    wordtoidfs = {}
   
    for key in documents.keys():
        pool.extend(documents[key])
        filetosetofwords[key] = set(documents[key])

    unique = set(pool)

    for word in unique:
        count = 0

        for key in filetosetofwords:
            if word in filetosetofwords[key]:
                count += 1

        wordtoidfs[word] = math.log(len(documents.keys())/ count)

    return wordtoidfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    score = {}
    for key in files.keys():
        score[key] = 0

    for word in query:
        for file in filetosetofwords:
            
            if word in filetosetofwords[file]:
                
                freq = 0
                
                for w in files[file]:
                    if w == word:
                        freq += 1

                score[file] += idfs[word] * freq
            
    return sorted(list(score.keys()),key = lambda x: score[x], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    score = {}
    word_measure = {}

    for key in sentences.keys():
        score[key] = 0
        word_measure[key] = 0

    for word in query:
        for sentence in sentences.keys():
        
            if word in sentences[sentence]:
                score[sentence] += idfs[word]
                word_measure[sentence] += 1

    for sentence in sentences.keys():
        word_measure[sentence] = word_measure[sentence] / len(sentence)
            

    ranked_sent = sorted(list(score.keys()),key = lambda x: (score[x], word_measure[x]), reverse=True)[:n]
    return ranked_sent


if __name__ == "__main__":
    main()
