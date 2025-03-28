DROP SCHEMA IF EXISTS scoring CASCADE;
CREATE SCHEMA scoring;

DROP TYPE IF EXISTS scoring.store_model_results CASCADE;
CREATE TYPE scoring.store_model_results AS (
  model_name TEXT,
  model_pickled BYTEA,
  exec_timestamp TIMESTAMP WITHOUT TIME ZONE,
  fit_duration REAL,
  n_features INTEGER,
  n_samples INTEGER,
  n_positive_samples INTEGER,
  auc REAL,
  aul REAL,
  scaling BYTEA
);

/*

Function to execute logistic regression and store resulting model.

PARAMS
------
  * model_name: name of the output model
  * features: serialized CSR matrix
  * labels: serialized np array

*/

DROP FUNCTION IF EXISTS scoring.logreg(model_name TEXT, features BYTEA[], labels BYTEA);
CREATE OR REPLACE FUNCTION scoring.logreg(model_name TEXT, features BYTEA[], labels BYTEA) RETURNS SETOF scoring.store_model_results AS $$
import numpy as np
import scipy.sparse as ss
import cPickle as pickle
import zlib
import base64
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn import metrics
from time import time
import datetime

def concatenate_csr_matrix_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    return ss.csr_matrix((new_data, new_indices, new_ind_ptr))

# loading features
for i in range(len(features)):
    mat = pickle.loads(zlib.decompress(base64.b64decode((features[i]))))
    if i == 0:
        X = mat
    else:
        X = concatenate_csr_matrix_by_columns(X, mat)

# loading labels
y = pickle.loads(zlib.decompress(base64.b64decode(labels)))

# scaling data
max_abs_scaler = MaxAbsScaler(copy=False)
X = max_abs_scaler.fit_transform(X)
# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# doing cross val
skf = StratifiedKFold(y=y_train, n_folds=5, shuffle=True, random_state=0)

# specifying model
Cs = np.array([1e-3,3e-3,5e-3,8e-3,1e-2,3e-2,5e-2,8e-2,1e-1])
clf = LogisticRegressionCV(Cs=Cs, 
                           class_weight='balanced',
                           cv=skf, scoring='roc_auc', 
                           penalty='l2', n_jobs=1, verbose=1, refit=True, max_iter = 300)

# fit
timestamp_current = datetime.datetime.utcnow()
t1 = time()
clf.fit(X_train, y_train)
t2 = time()

# metrics
pred = clf.predict_proba(X_test)[:,1]
auc = metrics.roc_auc_score(y_test, pred)
p = 1.0 * sum(y) / len(y)
aul = p / 2 + (1 - p) * auc

return([[model_name, 
         base64.b64encode(zlib.compress(pickle.dumps(clf, protocol = -1))), 
         timestamp_current, 
         t2-t1, 
         X.get_shape()[1],
         X.get_shape()[0],
         sum(y),
         auc, 
         aul,
         base64.b64encode(zlib.compress(pickle.dumps(max_abs_scaler, protocol = -1)))]])
$$ LANGUAGE plpythonu;

/*

Function to execute xgboost and store resulting model.

PARAMS
------
  * model_name: name of the output model
  * features: serialized CSR matrix
  * labels: serialized np array

*/

DROP FUNCTION IF EXISTS scoring.default_xgboost(model_name TEXT, features BYTEA[], labels BYTEA);
CREATE OR REPLACE FUNCTION scoring.default_xgboost(model_name TEXT, features BYTEA[], labels BYTEA) RETURNS SETOF scoring.store_model_results AS $$
import numpy as np
import scipy.sparse as ss
from math import sqrt
import cPickle as pickle
import zlib
import base64
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cross_validation import StratifiedKFold, train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from time import time
import datetime

def concatenate_csr_matrix_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    return ss.csr_matrix((new_data, new_indices, new_ind_ptr))

# loading features
for i in range(len(features)):
    mat = pickle.loads(zlib.decompress(base64.b64decode((features[i]))))
    if i == 0:
        X = mat
    else:
        X = concatenate_csr_matrix_by_columns(X, mat)

# loading labels        
y = pickle.loads(zlib.decompress(base64.b64decode(labels)))

# scaling data
max_abs_scaler = MaxAbsScaler(copy=False)
X = max_abs_scaler.fit_transform(X)
# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# doing cross val
skf = StratifiedKFold(y=y_train, n_folds=5, shuffle=True, random_state=0)

# specifying model
xgb_model = XGBClassifier()
parameters = {'silent': 1,
              'objective': 'binary:logistic',
              'learning_rate': 0.05,
              'gamma': 0,
              'max_depth': 6,
              'min_child_weight': 1.0/sqrt(1.0*sum(y)/len(y)),
              'max_delta_step': 1,
              'subsample': 0.8,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'scale_pos_weight': 1,
              'seed': 0}
xgb_model.set_params(**parameters)

# fit
current_time = datetime.datetime.utcnow()
t1 = time()
xgb_model.fit(X_train, y_train)
t2 = time()

# metrics
pred = xgb_model.predict_proba(X_test)[:,1]
auc = metrics.roc_auc_score(y_test, pred)
p = 1.0 * sum(y) / len(y)
aul = p / 2 + (1 - p) * auc

return([[model_name, 
         base64.b64encode(zlib.compress(pickle.dumps(xgb_model, protocol = -1))), 
         current_time, 
         t2-t1, 
         X.get_shape()[1],
         X.get_shape()[0],
         sum(y),
         auc, 
         aul,
         base64.b64encode(zlib.compress(pickle.dumps(max_abs_scaler, protocol = -1)))]])
$$ LANGUAGE plpythonu;

DROP FUNCTION IF EXISTS scoring.xgboost(model_name TEXT, features BYTEA[], labels BYTEA);
CREATE OR REPLACE FUNCTION scoring.xgboost(model_name TEXT, features BYTEA[], labels BYTEA) RETURNS SETOF scoring.store_model_results AS $$
import numpy as np
import scipy.sparse as ss
import scipy.stats as stats
from math import sqrt, floor, log10
import cPickle as pickle
import zlib
import base64
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn import grid_search
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from time import time
import datetime

def concatenate_csr_matrix_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    return ss.csr_matrix((new_data, new_indices, new_ind_ptr))

# loading features
for i in range(len(features)):
    mat = pickle.loads(zlib.decompress(base64.b64decode((features[i]))))
    if i == 0:
        X = mat
    else:
        X = concatenate_csr_matrix_by_columns(X, mat)

# loading labels        
y = pickle.loads(zlib.decompress(base64.b64decode(labels)))

# scaling data
max_abs_scaler = MaxAbsScaler(copy=False)
X = max_abs_scaler.fit_transform(X)
# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# doing cross val
skf = StratifiedKFold(y=y_train, n_folds=5, shuffle=True, random_state=0)

# specifying model
xgb_model = XGBClassifier()
parameters = {'silent': [1],
              'objective': ['binary:logistic'],
              'learning_rate': stats.uniform(0.01, 0.07),
              'gamma': [0, 1],
              'max_depth': [6, 7, 8, 9, 10],
              'min_child_weight': [1, 2, 3, 1.0/sqrt(1.0*sum(y)/len(y))],
              'max_delta_step': stats.randint(0, 10),
              'subsample': stats.uniform(0.3, 0.7),
              'colsample_bytree': stats.uniform(0.3, 0.5),
              'n_estimators': stats.randint(100, 500),
              'scale_pos_weight': [10**i for i in range(int(floor(log10(1.0*(len(y)-sum(y))/sum(y)))))],
              'seed': [0]}
clf = grid_search.RandomizedSearchCV(xgb_model, parameters, n_jobs=1, cv=skf, scoring='roc_auc', refit=True, n_iter=30)

# fit
current_time = datetime.datetime.utcnow()
t1 = time()
clf.fit(X_train, y_train)
t2 = time()

# metrics
xgb_model = clf.best_estimator_
pred = xgb_model.predict_proba(X_test)[:,1]
auc = metrics.roc_auc_score(y_test, pred)
p = 1.0 * sum(y) / len(y)
aul = p / 2 + (1 - p) * auc

return([[model_name, 
         base64.b64encode(zlib.compress(pickle.dumps(clf, protocol = -1))), 
         current_time, 
         t2-t1, 
         X.get_shape()[1],
         X.get_shape()[0],
         sum(y),
         auc, 
         aul,
         base64.b64encode(zlib.compress(pickle.dumps(max_abs_scaler, protocol = -1)))]])
$$ LANGUAGE plpythonu;