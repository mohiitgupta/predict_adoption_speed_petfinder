
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (7,7)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# In[2]:


def split_dataset(t_frac, random_state, dataset):
    testset=dataset.sample(frac=t_frac,random_state=random_state)
    trainset=dataset.drop(testset.index)
    testset.to_csv("testSet.csv", index = False)
    trainset.to_csv("trainingSet.csv", index = False)
    return trainset, testset

def get_features_labels(dataset):
    labels = dataset['AdoptionSpeed']
    features = dataset.drop(['AdoptionSpeed'], axis=1)
    return features, labels


# In[4]:


dataset = pd.read_csv("processed_trainingset.csv")


# In[5]:


trainset, testset = split_dataset(0.3, 47, dataset)
train_features_matrix, train_label_matrix = get_features_labels(trainset)
test_features, test_labels = get_features_labels(testset)


# In[6]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), fig_name='default_fig'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_micro')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(fig_name)
    return plt


# In[7]:


title = "Learning Curves"
n_jobs=-1
dtree_clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
svm_clf = SVC(gamma='auto')
nn_clf = MLPClassifier(early_stopping=True, solver='adam', alpha=1e-5, hidden_layer_sizes=(1500, 500), random_state=1, verbose = True)


# In[ ]:


# cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# plot_learning_curve(dtree_clf, title, train_features_matrix, train_label_matrix, cv=cv, n_jobs=n_jobs, fig_name='cross_validation_learning_curve_dtree')

# # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # # SVC is more expensive so we do a lower number of CV iterations:
# # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# # estimator = SVC(gamma=0.001)
# # plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

# # plt.show()
# plt.close()


# In[ ]:


# cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# plot_learning_curve(svm_clf, title, train_features_matrix, train_label_matrix,
#                     cv=cv, n_jobs=n_jobs, fig_name='cross_validation_learning_curve_svm')
# # plt.show()
# plt.close()


# In[ ]:


cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
plot_learning_curve(nn_clf, title, train_features_matrix, train_label_matrix,
                    cv=cv, n_jobs=n_jobs, fig_name='cross_validation_learning_curve_nn')
# plt.show()
plt.close()

