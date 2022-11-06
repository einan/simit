import pandas
import scipy.stats
import math

def get_jaccard_sim(str1, str2): 
    str1n = remove_punc(str1)
    str2n = remove_punc(str2)
    a = set(str1n.split()) 
    b = set(str2n.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def load_sts_dataset(filename):
    sent_pairs = []
    with open(filename, "r",encoding="utf8") as f:  
        for line in f:
            ts = line.strip().split("\t")      # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])

sts_test = load_sts_dataset('sts-test.csv')

test_list1 = sts_test['sent_1'].tolist()
test_list2 = sts_test['sent_2'].tolist()
test_scores = sts_test['sim'].tolist()

from similarity.cosine import Cosine
cos = Cosine(3)
cos_scores = []
for i in range(len(test_list1)):
    p0 = cos.get_profile(test_list1[i])
    p1 = cos.get_profile(test_list2[i])
    cos_scores.append(cos.similarity_profiles(p0, p1))
pearson_cos = scipy.stats.pearsonr(test_scores, cos_scores)
print('Pearson cos = {0}'.format(pearson_cos[0]*100))    

### n-gram calculations
from similarity.ngram import NGram
three_gram = NGram(3)
print('First sent={0}--second={1}--ngram score = {2}'.format(
    test_list1[0],test_list2[0],(1-three_gram.distance(test_list1[0], test_list1[1]))))
ngr_scores = []
for i in range(len(test_list1)):
    ngr_scores.append((1-three_gram.distance(test_list1[i],test_list2[i])))
pearson_ngr = scipy.stats.pearsonr(test_scores, ngr_scores)
print('Pearson n-gram = {0}'.format(pearson_ngr[0]*100))    
    

### jaccard calculations
from similarity.jaccard import Jaccard
jac = Jaccard(3)
jac_scores = []
for i in range(len(test_list1)):
    jac_scores.append(jac.similarity(test_list1[i],test_list2[i]))

pearson_jac = scipy.stats.pearsonr(test_scores, jac_scores)
print('Pearson jac = {0}'.format(pearson_jac[0]*100))

