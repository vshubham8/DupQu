print('---- Importing libraries ----')
import gensim
import numpy as np
import xgboost as xgb
from gc import collect
from scipy import sparse
from pprint import pprint
from pprint import pprint
from copy import deepcopy
from fuzzywuzzy import fuzz
from pandas import read_csv
from pandas import DataFrame
from nltk import word_tokenize
from easygui import fileopenbox
from sklearn import linear_model
from nltk.corpus import stopwords
from psutil import virtual_memory
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
# nltk.download('punkt')
# nltk.download('stopwords')
print('---- Libraries imported ----')

print('---- Loading data ----')
data_file = fileopenbox(msg = 'DATA FILE (Ex. train.csv)',title='Browse')
rows = int(input('No. of rows to take from data file: '))
data = read_csv(data_file, sep=',',nrows=rows)
data = data.drop(['id', 'qid1', 'qid2'], axis=1)
print('---- Data loaded ----')

print('---- Computing basic features ----')
# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions
data['diff_len'] = data.len_q1 - data.len_q2

# character length based features
data['len_char_q1'] = data.question1.apply(lambda x: 
len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: 
len(''.join(set(str(x).replace(' ', '')))))

# word length based features
data['len_word_q1'] = data.question1.apply(lambda x: 
len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: 
len(str(x).split()))

# common words in the two questions
data['common_words'] = data.apply(lambda x: 
len(set(str(x['question1'])
.lower().split())
.intersection(set(str(x['question2'])
.lower().split()))), axis=1)
fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 
        'len_char_q2', 'len_word_q1', 'len_word_q2',     
        'common_words']
pprint(fs_1)
print('---- Computed ----')

print('---- Computing fuzzy features ----')
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(
str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x: 
fuzz.partial_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:
fuzz.partial_token_set_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: 
fuzz.partial_token_sort_ratio(str(x['question1']), 
str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x: 
fuzz.token_set_ratio(str(x['question1']), 
str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: 
fuzz.token_sort_ratio(str(x['question1']), 
str(x['question2'])), axis=1)
fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
pprint(fs_2)
print('---- Computed ----')

print('---- Computing features for LSA (using SVD) ----')
tfv_q1 = TfidfVectorizer(min_df=3, 
max_features=None, 
strip_accents='unicode', 
analyzer='word', 
token_pattern=r'w{1,}',
ngram_range=(1, 2), 
use_idf=1, 
smooth_idf=1, 
sublinear_tf=1,
stop_words='english')

tfv_q2 = deepcopy(tfv_q1)

q1_tfidf = tfv_q1.fit_transform(data.question1.fillna(""))
q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))

svd_q1 = TruncatedSVD(n_components=1)
svd_q2 = TruncatedSVD(n_components=1)

question1_vectors = svd_q1.fit_transform(q1_tfidf)
question2_vectors = svd_q2.fit_transform(q2_tfidf)

# obtain features by stacking the sparse matrices together
fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))

tfv = TfidfVectorizer(min_df=3, 
                      max_features=None, 
                      strip_accents='unicode', 
                      analyzer='word', 
                      token_pattern=r'w{1,}',
                      ngram_range=(1, 2), 
                      use_idf=1, 
                      smooth_idf=1, 
                      sublinear_tf=1,
                      stop_words='english')

# combine questions and calculate tf-idf
q1q2 = data.question1.fillna("") 
q1q2 += " " + data.question2.fillna("")
fs3_2 = tfv.fit_transform(q1q2)

# obtain features by stacking the matrices together
fs3_3 = np.hstack((question1_vectors, question2_vectors))

# obtain features by stacking the matrices together
x = sparse.hstack((q1_tfidf,q2_tfidf))
svd_q1q2 = TruncatedSVD(n_components=1)
xx = svd_q1q2.fit_transform(x)
fs3_4 = xx

# obtain features by stacking the matrices together
x = sparse.hstack((q1_tfidf,q2_tfidf))
svd_ff = TruncatedSVD(n_components=1)
xx = svd_ff.fit_transform(fs3_2)
fs3_5 = xx
print('---- Computed ----')

print('---- Computing Word2Vec features ----')
print('\t---- Loading (google-news) glove vectors ----')
glove_vectors_file = fileopenbox(msg = 'Word2Vec FILE (Ex. xyz.bin.gz)',title='Browse')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print('\t---- Model loaded ----')

# Ex.
# model['hello']

stop_words = set(stopwords.words('english'))
def sent2vec(s, model): 
    M = []
    words = word_tokenize(str(s).lower())
    for word in words:
        #It shouldn't be a stopword, nor contain numbers, and be part of word2vec
        if word not in stop_words and word.isalpha() and word in model:
            M.append(model[word])
    M = np.array(M)
    if len(M) > 0:
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    else:
        return np.zeros(300)

w2v_q1 = np.array([sent2vec(q, model) 
                   for q in data.question1])
w2v_q2 = np.array([sent2vec(q, model) 
                   for q in data.question2])

data['cosine_distance'] = [cosine(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['cityblock_distance'] = [cityblock(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['canberra_distance'] = [canberra(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['euclidean_distance'] = [euclidean(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['minkowski_distance'] = [minkowski(x,y,3) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['braycurtis_distance'] = [braycurtis(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]

fs4_1 = ['cosine_distance', 'cityblock_distance', 
         'jaccard_distance', 'canberra_distance', 
         'euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']

w2v = np.hstack((w2v_q1, w2v_q2))

def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

print('\t---- Calculating WMD ----')
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], model), axis=1)
model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], model), axis=1)
fs4_2 = ['wmd', 'norm_wmd']
print('\t---- Calculated ----')

pprint(fs4_1)
print('---- Computed ----')

print('---- Deleting variables that are not required ----')
del([tfv_q1, tfv_q2, tfv, q1q2, 
     question1_vectors, question2_vectors, svd_q1, 
     svd_q2, q1_tfidf, q2_tfidf])
del([w2v_q1, w2v_q2])
del([model])
collect()
virtual_memory()
print('---- Deleted ----')

s='''
At this point, we simply recap the different features created up to now, and their meaning in terms of generated features:

fs_1: List of basic features
fs_2: List of fuzzy features
fs3_1: Sparse data matrix of TFIDF for separated questions
fs3_2: Sparse data matrix of TFIDF for combined questions
fs3_3: Sparse data matrix of SVD
fs3_4: List of SVD statistics
fs4_1: List of w2vec distances
fs4_2: List of wmd distances
w2v: A matrix of transformed phrase’s Word2vec vectors by means of the Sent2Vec function
'''
print(s)

print('---- Making Logistic Regression model ----')
scaler = StandardScaler()
y = data.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
# X = data[fs_1+fs_2+fs3_4+fs4_1+fs4_2]
X = data[fs_1+fs_2+fs4_1+fs4_2]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)
X = np.hstack((X, fs3_3))

np.random.seed(42)
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)
n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
x_train = X[idx_train]
y_train = np.ravel(y[idx_train])
x_val = X[idx_val]
y_val = np.ravel(y[idx_val])

logres = linear_model.LogisticRegression(C=0.1, solver='sag', max_iter=1000)
print('\t- Training')
logres.fit(x_train, y_train)
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("\t- Accuracy:" ,round(log_res_accuracy*100.0,2),'%')

test_ids = idx[:n_split]
q1s = data.iloc[test_ids]['question1']
q2s = data.iloc[test_ids]['question2']

params = dict()
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['logloss', 'error']
params['eta'] = 0.02
params['max_depth'] = 4
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_val, label=y_val)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 5000, watchlist, 
                early_stopping_rounds=50, verbose_eval=False)
xgb_preds = (bst.predict(d_valid) >= 0.5).astype(int)
percents = bst.predict(d_valid)
xgb_accuracy = np.sum(xgb_preds == y_val) / len(y_val)
print("\t- Xgb accuracy:",round(xgb_accuracy*100.0,2),'%')

roundoff = []
for i in percents:
    roundoff.append(round(i*100.0,2))
pprint(roundoff)

s=[]
l = 0
for i in test_ids:
    s.append([str(q1s[i]),str(q2s[i]),str(roundoff[l])])
    l+=1
df=DataFrame(s,columns=['Q1','Q2','prediction(in %)'])
df.to_csv('results.csv',sep=',',index=False)
print('\t- Results are saved in \'results.csv\'')
print('---- DONE ----')