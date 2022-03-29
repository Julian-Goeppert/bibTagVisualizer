from bs4 import BeautifulSoup
import csv
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def htmlTagSearch(htmlFileName):
    fp = open(htmlFileName)
    soup = BeautifulSoup(fp, "html.parser")
    res = soup.find_all("ul", class_ ="tags")
    return res

def tagSorting(htmlResult):
    tagsPerPaper = []
    for line in htmlResult:
        tags = []
        for buzzword in line:
            string = buzzword.string
            if(string != '\n'):
                tags.append(string)
        tagsPerPaper.append(tags)
    return tagsPerPaper

def csv_exporter(csv_name, list):
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for row in list:
            writer.writerow(row)

def coocurranceMatrixCreator(listedList):
    cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english') # You can define your own parameters
    text = []
    for stringList in listedList:
        for string in stringList:
            text.append(string)
    X = cv.fit_transform(text)
    Xc = (X.T*X)
    Xc.setdiag(0)
    names = cv.get_feature_names_out()
    df = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)

    return df, text
    # found on https://www.pingshiuanchua.com/blog/post/keyword-network-analysis-with-python-and-gephi

def visualizeCoocurranceMatrix(df, characters):
    #found on http://andrewtrick.com/stormlight_network.html
    edge_list = [] #test networkx
    for index, row in df.iterrows():
        i = 0
        for col in row:
            weight = float(col)/464
            edge_list.append((index, df.columns[i], weight))
            i += 1
    #Remove edge if 0.0
    updated_edge_list = [x for x in edge_list if not x[2] == 0.0]

    #create duple of char, occurance in novel
    node_list = []
    for i in characters:
        for e in updated_edge_list:
            if i == e[0] and i == e[1]:
                node_list.append((i, e[2]*6))
    for i in node_list:
        if i[1] == 0.0:
            node_list.remove(i)

    #remove self references
    for i in updated_edge_list:
        if i[0] == i[1]:
            updated_edge_list.remove(i)

            #set canvas size
    plt.subplots(figsize=(14,14))
    #networkx graph time!
    G = nx.Graph()
    for i in sorted(node_list):
        G.add_node(i[0], size = i[1])
    G.add_weighted_edges_from(updated_edge_list)

    node_order = characters

    #reorder node list
    updated_node_order = []
    for i in node_order:
        for x in node_list:
            if x[0] == i:
                updated_node_order.append(x)

    #reorder edge list - this was a pain
    test = nx.get_edge_attributes(G, 'weight')
    updated_again_edges = []
    for i in nx.edges(G):
        for x in test.keys():
            if i[0] == x[0] and i[1] == x[1]:
                updated_again_edges.append(test[x])

    #drawing custimization
    node_scalar = 800
    edge_scalar = 10
    sizes = [x[1]*node_scalar for x in updated_node_order]
    widths = [x*edge_scalar for x in updated_again_edges]

    #draw the graph
    pos = nx.spring_layout(G, k=0.42, iterations=17)

    nx.draw(G, pos, with_labels=True, font_size = 8, font_weight = 'bold',
        node_size = sizes, width = widths)

    plt.savefig("examples/network_graph.png") # save as png

def main():
    res = htmlTagSearch("examples/Zotero-Bericht.htm")
    tags = tagSorting(res)
    csv_exporter("examples/results.csv", tags)
    df, text = coocurranceMatrixCreator(tags)
    visualizeCoocurranceMatrix(df, text)


if __name__ == "__main__":
    main()
