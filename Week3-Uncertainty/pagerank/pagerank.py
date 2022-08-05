import os
from pydoc import doc
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    model = dict()

    s = len(corpus)
    links = corpus[page]
    lsize = len(links)

    for page in corpus:

        link_p = float()
        rand_p = float()

        if page in links:
            link_p = damping_factor/lsize
            #print(f"link_p = {link_p}")

        rand_p = (1 - damping_factor)/s
        #print(f"rand_p = {rand_p}")

        #print(f"p = {link_p + rand_p}")

        model[page] = link_p + rand_p 

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    rank = dict()

    for page in corpus:
        rank[page] = 0

    pages = list(corpus.keys())
    page = random.choice(pages)

    """
    page = '2.html'
    dist = transition_model(corpus, page, damping_factor)
    print(dist)
    
    """
    
    for i in range(n):
        rank[page] += 1

        dist = transition_model(corpus, page, damping_factor)
        w = []

        for p in dist:
            w.append(dist[p])

        page = random.choices(population=pages, weights=w, k=1)[0]
        

    for page in rank:
        rank[page] = rank[page] / n

    return rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    rank = dict()
    size = len(corpus)

    for page in corpus:
        rank[page] = 1/size

    diff = float(1)

    revcorpus = genIncoming(corpus)

    while diff >= 0.001:
        diff = calc(corpus, revcorpus, rank, damping_factor)

    return rank


def genIncoming(corpus):

    revcorpus = dict()

    for page in corpus:
        revcorpus[page] = set()

    for page in corpus:
        for p in corpus[page]:
            revcorpus[p].add(page)

    return revcorpus


def calc(corpus, revcorpus, rank, d):

    size = len(corpus)
    k = (1-d)/size
    diff = float(0)
    prevrank = rank.copy()

    for page in corpus:

        sum = float(0)

        for p in revcorpus[page]:
            num_links = len(corpus[p])
            sum += prevrank[p]/num_links

        for p in corpus:
            if not corpus[p]:
                sum += prevrank[p]/ len(corpus)
        
        newrank = k + (d * sum)
        oldrank = prevrank[page]

        diff = max(diff, newrank - oldrank)
        rank[page] = newrank

    return diff

if __name__ == "__main__":
    main()
