import os
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
    distribution = {}
    links = corpus[page]
    pages = len(corpus)
    d = corpus.copy()

    if links == {}:
        for link in corpus:
            distribution[link] = 1 / pages
        return corpus

    for page in links:
        distribution[page] = damping_factor / len(links)
        
    for page in corpus:
        if page in links:
            distribution[page] += ((1 - damping_factor) / len(corpus))
        else:
            distribution[page] = ((1 - damping_factor) / len(corpus))
    
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}

    # starting page
    page = random.choice(list(corpus))

    # iterate over samples
    for i in range(n - 1):
        prob = transition_model(corpus, page, damping_factor)
        
        page = random.choices(list(prob), weights=prob.values(), k=1).pop()

        if page in pagerank:
            pagerank[page] += 1
        else:
            pagerank[page] = 1

    # find the pagerank
    for page in pagerank:
        pagerank[page] = pagerank[page] / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = len(corpus)

    d = {}
    
    for page in corpus:
        d[page] = 1/len(corpus)
    not_convergent = True
    while not_convergent:
        d_copy = d.copy()
        d_diff = {}
        for page in corpus.keys():
            prob = 0
            
            for page_i, pages in corpus.items():
                if page in pages:
                    prob += (d_copy[page_i] / len(pages))
                elif len(pages) == 0:
                    prob += 1 / len(corpus)
            d[page] = (1 - damping_factor) / len(corpus) + (damping_factor * prob)
            d_diff[page] = abs(d_copy[page] - d[page])
        not_convergent = False
        for page in d_diff:
            if d_diff[page] > 0.001:
                not_convergent = True
    
    sum_pagerank = 0
    for k in d:
        sum_pagerank += d[k]

    for k in d:
        d[k] = d[k] / sum_pagerank

    return d


if __name__ == "__main__":
    main()
