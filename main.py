# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Common imports
import numpy as np
import os

import pandas as pd

def load_housing_data(housing_path="."):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()


#分层抽样
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


#删除median_house_value,此时housing被训练集替换了
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

housingt = strat_test_set.drop("median_house_value", axis=1)
housingt_labels = strat_test_set["median_house_value"].copy()

#处理缺失值，使用中位数填充
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num) #算均值，方差，最大值，最小值供transform使用
X = imputer.transform(housing_num)#在fit的基础上，进行imputer变换操作
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

housingt_num = housingt.drop("ocean_proximity", axis=1)
imputer.fit(housingt_num)
X = imputer.transform(housingt_num)
housing_tr = pd.DataFrame(X, columns=housingt_num.columns,
                          index=housingt_num.index)


#使用OneHotEncoder方法处理object类型
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()

#添加三个新属性
from sklearn.base import BaseEstimator, TransformerMixin
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)


#pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

housingt_num_tr = num_pipeline.fit_transform(housingt_num)


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

housingt_prepared = full_pipeline.fit_transform(housingt)



#模型训练
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print("svm_rmse =",svm_rmse)


#交叉验证
from sklearn.model_selection import cross_val_score

scores = cross_val_score(svm_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-scores)

print("svm_rmse CV mean =",svm_rmse_scores.mean())


# #网格搜索调参
# from sklearn.model_selection import GridSearchCV

# param_grid = [
#         {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
#         {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
#          'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
#     ]

# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
# grid_search.fit(housing_prepared, housing_labels)

# negative_mse = grid_search.best_score_
# gridSearchCV_rmse = np.sqrt(-negative_mse)
# print("gridSearchCV_rmse = ")
# print(gridSearchCV_rmse)

# print("grid_search.best_params_ = ")
# print(grid_search.best_params_)



# #随机搜索调参
from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import expon, reciprocal

# # see https://docs.scipy.org/doc/scipy/reference/stats.html
# # for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# # Note: gamma is ignored when kernel is "linear"
# param_distribs = {
#         'kernel': ['linear', 'rbf'],
#         'C': reciprocal(20, 200000),
#         'gamma': expon(scale=1.0),
#     }

# svm_reg = SVR()
# rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
#                                 n_iter=50, cv=5, scoring='neg_mean_squared_error',
#                                 verbose=2, random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)

# negative_mse = rnd_search.best_score_
# random_rmse = np.sqrt(-negative_mse)
# print("random_rmse = ")
# print(random_rmse)

# print("rnd_search.best_params_ = ")
# print(rnd_search.best_params_)


#使用随机搜索的最佳参数在测试集上的rmse
params_ = {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}
svm_reg_new = SVR(**params_)
svm_reg_new.fit(housing_prepared, housing_labels)
housingt_predictions = svm_reg_new.predict(housingt_prepared)
svm_mse_final = mean_squared_error(housingt_labels, housingt_predictions)
svm_rmse_final = np.sqrt(svm_mse_final)
print("svm_rmse_final =",svm_rmse_final)