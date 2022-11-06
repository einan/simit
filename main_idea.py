import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
doc = nlp('A woman picks up and holds a baby kangaroo')
for token in doc:
    print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
    displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
    
import matplotlib.pyplot as plt
import networkx as nx

edges = []
for token in doc:
    for child in token.children:
        edges.append(('{0}_{1}'.format(token.lower_,token.dep_),'{0}_{1}'.format(child.lower_,child.dep_)))

graph = nx.Graph(edges)


nx.draw(graph, with_labels=True, width=1,alpha=0.9, node_size=800, node_color="skyblue", font_weight='bold')
plt.show()


dict_bfs = dict(nx.bfs_successors(graph,"picks_ROOT"))
list_keys = [k for k in dict_bfs]
list_values = [ v for v in dict_bfs.values()]
valst = []
for i in range(len(list_values)):
    tmp = list_values[i]
    if list_keys[i] not in valst:
        valst.append(list_keys[i])
    for j in range(len(tmp)):
        valst.append(tmp[j])
print(valst)


T = nx.depth_first_search.dfs_tree(graph,"picks_ROOT")

plt.figure()
nx.draw(T, with_labels=True)
plt.show()

bfs_w_list = []
for i in range(len(valst)):
    x = valst[i].split("-")
    bfs_w_list.append(x[0])
print(bfs_w_list)


import pandas

def load_sts_dataset(filename):
    sent_pairs = []
    with open(filename, "r",encoding="utf8") as f:  
        for line in f:
            ts = line.strip().split("\t")      # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])

sts_train = load_sts_dataset('sts-train.csv')

train_list1 = sts_train['sent_1'].tolist()
train_list2 = sts_train['sent_2'].tolist()
train_scores = sts_train['sim'].tolist()


import gensim
from gensim.utils import simple_preprocess

# generates the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(simple_preprocess(list_of_words), [i])

train_data = list(create_tagged_document(train_list1))

print(train_data[:1])

from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api

# Download dataset
#dataset = api.load("text8")
#data = [d for d in dataset]

# Split the data into 2 parts. Part 2 will be used later to update the model
#data_part1 = data[:1000]

word_list = []
for i, list_of_words in enumerate(train_list1):
    word_list.append(simple_preprocess(train_list1[i]))
    
print(word_list[0])

#data_part2 = data[1000:]

# trains Word2Vec model (Defaults result vector size = 100)
model = Word2Vec(word_list, min_count = 0, workers=cpu_count())

# gets the word vector for given word
model['plane']
#> array([ 0.0512,  0.2555,  0.9393, ... ,-0.5669,  0.6737], dtype=float32)

model.most_similar('plane')
#> [('discussion', 0.7590423822402954),
#>  ('consensus', 0.7253159284591675),
#>  ('discussions', 0.7252693176269531),
#>  ('interpretation', 0.7196053266525269),
#>  ('viewpoint', 0.7053568959236145),
#>  ('speculation', 0.7021505832672119),
#>  ('discourse', 0.7001898884773254),
#>  ('opinions', 0.6993060111999512),
#>  ('focus', 0.6959210634231567),
#>  ('scholarly', 0.6884037256240845)]

# saves and loads the model
model.save('newmodel')
model = Word2Vec.load('newmodel')

print(model['plane'])

word_vectors = model.wv

from gensim.matutils import softcossim
from gensim import corpora

sent_1 = 'A plane is taking off'.split()
sent_2 = 'A man is playing a flute'.split()
sent_3 = 'A man is spreading shreded cheese on a pizza'.split()

# Prepare a dictionary and a corpus.
documents = [sent_1, sent_2, sent_3]
dictionary = corpora.Dictionary(documents)

# Prepare the similarity matrix
similarity_matrix = word_vectors.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# Convert the sentences into bag-of-words vectors.
sent_1 = dictionary.doc2bow(sent_1)
sent_2 = dictionary.doc2bow(sent_2)
sent_3 = dictionary.doc2bow(sent_3)

# Compute soft cosine similarity
print(softcossim(sent_1, sent_2, similarity_matrix))
#> 0.7868705819999783

print(softcossim(sent_1, sent_3, similarity_matrix))
#> 0.6036445529268666

print(softcossim(sent_2, sent_3, similarity_matrix))
#> 0.60965453519611
