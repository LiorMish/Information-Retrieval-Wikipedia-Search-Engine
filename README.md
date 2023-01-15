# information_retrieval - Search Engine
In this project we build a search engine for the entire english Wikipedia corpus.
All the search functions described in the search_frontend.py file.
The backend of the whole process is in search_engine.py file.



# Main functions:


# Search:
This is the main search function that returns up to a 100 of our best search results for the query.
In this part we are using BM25 similarity measure on the title and body indices and merging their scores together.
We tried many different models as Word2Vec query expansions, stemming and others and chosed the best search engine we find.

# Search body:
Returns up to a 100 search results for the query using TFIDF AND Csine Similarity of the body articles.

# Search title:
Returns all search results that contain a query word in the title of articles, ordered in descending order of the number of query words that appear in the title. For example, a document with a title that matches two of the query words will be ranked before a document with a title that matches only one query term.

# Search anchor:
Returns all search results that contain a query word in the anchor text of articles, ordered in descending order of the number of query words that appear in the text linking to the page. For example, a document with an anchor text that matches two of the query words will be ranked before a document with anchor text that matches only one query term.

# Get page_rank:
Returns PageRank values for a list of provided wiki article IDs.

# Get page_view:
Returns the number of page views that each of the provide wiki articles had in August 2021.
