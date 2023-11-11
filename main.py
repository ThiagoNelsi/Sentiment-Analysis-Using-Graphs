import re
import nltk, numpy, string
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def preprocess(text):
    pattern = r"<[^>]+>"

    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if not bool(re.search(pattern, token))]

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords_removed = [token for token in tokens if token not in stopwords]

    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in stopwords_removed]

    stemmer = nltk.stem.PorterStemmer()
    stems = [stemmer.stem(token) for token in stopwords_removed]

    return lemmas, stems

def create_word_graph(tokens, word_frame):
    graph = nx.Graph()
    graph.add_nodes_from(tokens)

    for i in range(len(tokens) - 1):
        for j in range(word_frame - 1):
            if i + j + 1 < len(tokens):
                graph.add_edge(tokens[i], tokens[i + j + 1])


    return graph

def draw_graph(graph, filename):
    net = Network(notebook=True)
    net.from_nx(graph)
    net.show(filename)

def main():
    data = pd.read_csv('movie.csv', nrows=1000)
    array = data.values

    positive_graph = nx.Graph()
    negative_graph = nx.Graph()

    cont = 0
    for row in array:
        cont += 1
        print(f'{cont}/{len(array)} - {"Positive" if row[1] else "Negative"}')
        lemmas, stems = preprocess(row[0])
        graph = create_word_graph(stems, 3)

        if row[1] == 1:
            positive_graph = nx.compose(positive_graph, graph)
        else:
            negative_graph = nx.compose(negative_graph, graph)

    nx.write_graphml(positive_graph, 'positive.graphml')
    nx.write_graphml(negative_graph, 'negative.graphml')

main()