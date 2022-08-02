from feature_selection import lfs, ufs_sp_l_2_1, ufs_sp_f, mrmr_score, relief_f

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


named_score_funcs = {'lfs': lfs,
                     'ufs_sp_l_2_1': ufs_sp_l_2_1,
                     'ufs_sp_f': ufs_sp_f,
                     'mrmr_score': mrmr_score,
                     'relief_f': relief_f}
score_funcs = list(named_score_funcs.values())

named_classifiers = {'knn': KNeighborsClassifier(),
                     'nb': GaussianNB(),
                     'lr': LogisticRegression(),
                     'svc': SVC(),
                     'rf': RandomForestClassifier()}
classifiers = list(named_classifiers.values())

ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
