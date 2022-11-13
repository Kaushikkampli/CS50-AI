import nltk
from nltk.tree import *
from nltk.tokenize import word_tokenize
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP | VP |S Conj S
NP -> N | NP VP | Det NP | P NP | Adj NP | Adv NP | NP Adv | Conj NP
VP -> V | VP NP | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    words = word_tokenize(sentence.lower())
    
    temp = words.copy()
    
    for word in temp:
        if not word.isalpha():
            words.remove(word) 
        
    return words

def checkSubNounPhrase(tree):

    try :
        if tree.label() == "NP":
            return True
    except :
        return False

    for subtree in tree:
        return checkSubNounPhrase(subtree)
        
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    chunks = []

    subexists = checkSubNounPhrase(tree)

    if len(tree) == 1 and tree.label() == "NP":
        chunks.append(tree)
        return chunks

    if tree.label() == "NP" and not subexists:
        chunks.append(tree)
        return chunks

    for subtree in tree:

        node = subtree.label()
        if node == "NP" or node == "VP" or node == "S":
            subsubtree = np_chunk(subtree)
            for np in subsubtree:
                chunks.append(np)

    return chunks


if __name__ == "__main__":
    main()
