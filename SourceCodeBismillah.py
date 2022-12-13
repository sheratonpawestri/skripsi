# membaca dataset
import numpy as np
import pandas as pd

dataset = pd.read_excel("datasetPilpres_ver2.xlsx")

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import re
# membuat stopword remover
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
# membuat stemmer
factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

# PRAPEMROSESAN
documents = []
for sen in range(0, len(dataset['Teks'])):
    document = re.sub(r'\W', ' ', str(dataset['Teks'][sen])) # menghilangkan spesial karakter
    document = re.sub(r'\s+', ' ', document, flags=re.I) # menghilangkan spasi double
    document = document.lower() # casefolding
    document = word_tokenize(document) # tokenisasi
    document = [stopword.remove(word) for word in document] # stopword
    document = [stemmer.stem(word) for word in document] # stem
    document = ' '.join(document)
    documents.append(document)

# TF IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(documents)
tfidfconverter = TfidfTransformer(smooth_idf=True,use_idf=True)

X = tfidfconverter.fit_transform(vec)
y = dataset.iloc[:,1]

df_X = pd.DataFrame(X)
df_X.to_excel (r'I:/[SKRIPSI]/ZZ COBA/dataX.xlsx', index = False, header=True)
datasetxy = {'X':df_X[0],'y':y}
df_datasetxy = pd.DataFrame(datasetxy)
df_datasetxy.to_excel (r'I:/[SKRIPSI]/ZZ COBA/datasetxy.xlsx', index = False, header=True)

features = vectorizer.get_feature_names()

# KLASIFIKASI SVM dan K-fold Cross Validation
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

C_array = [0.001, 1, 1000] 
cm_total = []
scores_total = [] 
for c in range(len(C_array)):
    clf = SVC(kernel='linear', C = C_array[c])
    cv = KFold(n_splits=10, shuffle=False)
    scores = []
    cm = []
    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n", "Test Index: ", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Training
        import time
        start_training = time.time()
        clf.fit(X_train,y_train) 
        end_training = time.time()
        training_time = end_training-start_training
        print("Nilai C= ", C_array[c] , "\n", "waktu training = ", training_time)
        
        # Testing
        start_testing = time.time()
        y_predict = clf.predict(X_test)
        end_testing = time.time()
        testing_time = end_testing-start_testing
        print("Nilai C= ", C_array[c] , "\n", "waktu testing = ", testing_time)
        
        scores.append(clf.score(X_test, y_test))
        cm.append(confusion_matrix(y_test, y_predict))
    scores_total.append(scores)
    cm_total.append(cm)
#akurasi_SVM = np.mean(scores_total)
# PSO objective function
import pyswarms as ps
# Objective Function
def f_per_particle(m, alpha):

    total_features = X.shape[1]
    
    # subset feature dari binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    
    # Klasifikasi dan menentukan P
    clf.fit(X_subset, y)
    P = (clf.predict(X_subset) == y).mean()
    
    # Menghitung objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(X, alpha=0.88):

    n_particles = X.shape[0]
    j = [f_per_particle(X[i], alpha) for i in range(n_particles)]
    return np.array(j)

# PSO Inisialisasi swarm
options = {'c1': 2, 'c2': 2, 'w': 1, 'k': 100 , 'p':2}

dimensions = X.shape[1] # dimensions = jumlah fitur
optimizer = ps.discrete.BinaryPSO(n_particles=100, dimensions=dimensions, options=options)
cost, pos = optimizer.optimize(f, iters=200)

# SELEKSI FITUR
X_selected_features = X[:,pos == 1]

# Performa klasifikasi dan simpan performa di P
clf.fit(X_selected_features, y)
# Menghitung Performa PSO
subset_performance = (clf.predict(X_selected_features) == y).mean()
print("================== PERFORMA PSO ====================")
print('Performa subset fitur: %.3f' % (subset_performance))

best_position = optimizer.swarm.best_pos

df_X_Selected_features = pd.DataFrame(X_selected_features)
df_X_Selected_features.to_excel(r'I:/[SKRIPSI]/ZZ COBA/dataX_Selected_feature.xlsx', index = False, header=True)

datasetseleksi = {'eX':df_X_Selected_features[0],'Ye':y}
df_datasetseleksi = pd.DataFrame(datasetseleksi)
df_datasetseleksi.to_excel(r'I:/[SKRIPSI]/ZZ COBA/datasetseleksi.xlsx', index = False, header=True)

cm_total_PSO = []
scores_total_PSO = []
for c in range(len(C_array)):
    clf = SVC(kernel='linear', C = C_array[c])
    cv = KFold(n_splits=10, shuffle=False)
    scores_PSO = []
    cm_PSO = []
    for train_index, test_index in cv.split(X_selected_features):
        #print("Train Index: ", train_index, "\n", "Test Index: ", test_index)
        
        X_train_PSO, X_test_PSO = X_selected_features[train_index], X_selected_features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Training
        start_training_PSO = time.time()
        clf.fit(X_train_PSO,y_train) 
        end_training_PSO = time.time()
        training_time_PSO = end_training_PSO - start_training_PSO
        print("Nilai C= ", C_array[c] , "\n", "waktu training = ", training_time_PSO)
     
        # Testing
        y_predict_PSO = []
        start_testing_PSO = time.time()
        y_predict_PSO = clf.predict(X_test_PSO)
        end_testing_PSO = time.time()
        testing_time_PSO = end_testing_PSO - start_testing_PSO
        print("Nilai C= ", C_array[c] , "\n", "waktu testing = ", testing_time_PSO)
    
        scores_PSO.append(clf.score(X_test_PSO, y_test))
        cm_PSO.append(confusion_matrix(y_test, y_predict_PSO))
    scores_total_PSO.append(scores_PSO)
    cm_total_PSO.append(cm_PSO)

data_scores = {'SVM C = 0.001':scores_total[0],
               'SVM C = 1':scores_total[1],
               'SVM C = 1000':scores_total[2],
               'SVM PSO C = 0.001':scores_total_PSO[0],
               'SVM PSO C = 1':scores_total_PSO[1],
               'SVM PSO C = 1000':scores_total_PSO[2]}
data_confussionmatrix = {'SVM C = 0.001':cm_total[0],
                         'SVM C = 1':cm_total[1],
                         'SVM C = 1000':cm_total[2],
                         'SVM PSO C = 0.001':cm_total_PSO[0],
                         'SVM PSO C = 1':cm_total_PSO[1],
                         'SVM PSO C = 1000':cm_total_PSO[2]}
df_scores = pd.DataFrame(data_scores)
df_confussionmatrix = pd.DataFrame(data_confussionmatrix)
df_scores.to_excel (r'I:/[SKRIPSI]/ZZ COBA/scores.xlsx', index = False, header=True)
df_confussionmatrix.to_json (r'I:/[SKRIPSI]/ZZ COBA/confussionmatrix.json')

# Import modules
import matplotlib.pyplot as plt
# Import PySwarms
from pyswarms.utils.plotters import plot_cost_history
plot_cost_history(cost_history=optimizer.cost_history)
plt.show()