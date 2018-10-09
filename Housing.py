import os
import pandas as pd
import tarfile
import matplotlib.pyplot as plt
import hashlib
import numpy as np

from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix

from CombinedAttributesAdder import CombinedAttributesAdder
from DataFrameSelector import DataFrameSelector
from LabelBinarizerPipelineFriendly import LabelBinarizerPipelineFriendly

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_CSV = "housing.csv"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def set_print_options():
    pd.set_option('display.width', 500)
    pd.set_option('precision', 3)


def print_with_header(header='', body=''):
    print(header)
    if callable(body):
        body()
    else:
        print(body)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    tgz_file = tarfile.open(tgz_path)
    tgz_file.extractall(housing_path)
    tgz_file.close()

    os.remove(tgz_path)


def load_housing_data(housing_path=HOUSING_PATH, housing_csv=HOUSING_CSV):
    housing_dataset = os.path.join(housing_path, housing_csv)
    if not os.path.isfile(housing_dataset):
        print("housing.csv is not found at %s" % housing_path)
        fetch_housing_data()

    return pd.read_csv(housing_dataset)


def test_set_check(identifier, test_set_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_set_ratio


def split_train_test(dataset, test_set_ratio, identifier, hash=hashlib.md5):
    ids = dataset[identifier]
    test_set_ids = ids.apply(lambda id: test_set_check(id, test_set_ratio, hash))
    return dataset[~test_set_ids], dataset[test_set_ids]


def stratified_split_train_test_with_income(dataset, test_set_ratio, random_state):
    stratified_split = StratifiedShuffleSplit(n_splits=1, random_state=random_state, test_size=test_set_ratio)
    dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)
    dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)

    # print(housing_data["income_cat"].value_counts() / len(housing_data))

    for train_index, test_index in stratified_split.split(dataset, dataset["income_cat"]):
        strata_train_set, strata_test_set = dataset.loc[train_index], dataset.loc[test_index]

    for dataset in strata_train_set, strata_test_set:
        dataset.drop(["income_cat"], axis=1, inplace=True)

    return strata_train_set, strata_test_set


if '__main__' == __name__:
    set_print_options()
    housing_data = load_housing_data()
    columns_names = list(housing_data)

    encoder = LabelEncoder()
    housing_data_cat_feature = housing_data["ocean_proximity"]
    housing_data_cat_feature_encoded = encoder.fit_transform(housing_data_cat_feature)
    print_with_header("Ocean Proximity Encoded", housing_data_cat_feature_encoded)
    print_with_header("Classes Encoded", encoder.classes_)

    encoder = OneHotEncoder()
    housing_data_cat_feature_1hot_encoded = encoder.fit_transform(housing_data_cat_feature_encoded.reshape(-1, 1))
    print_with_header("Ocean Proximity OneHot Encoded", housing_data_cat_feature_1hot_encoded.toarray())

    encoder = LabelBinarizer()
    housing_data_cat_feature_1hot_encoded = encoder.fit_transform(housing_data_cat_feature)
    print_with_header("Ocean Proximity 1Hot Encoded with LabelBinarizer",
                      housing_data_cat_feature_1hot_encoded)

    attrs_addr = CombinedAttributesAdder()
    housing_data_attrs = attrs_addr.transform(housing_data.values)
    columns_names = columns_names + ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    print_with_header("Columns Names", columns_names)
    # housing_data = pd.DataFrame(data=housing_data_attrs, columns=columns_names)

    housing_data["rooms_per_household"] = housing_data["total_rooms"] / housing_data["households"]
    housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
    housing_data["population_per_household"] = housing_data["population"] / housing_data["households"]

    print(housing_data['ocean_proximity'].value_counts())
    print_with_header("===== Peek =====", housing_data.head())
    print_with_header("===== Info =====", housing_data.info)
    print_with_header("=== Describe ===", housing_data.describe())
    print_with_header("===== Corr =====", housing_data.corr())
    print_with_header("===== Corr of Median House Values =====",
                      housing_data.corr()["median_house_value"].sort_values(ascending=True))
    print_with_header("===== Skew =====", housing_data.skew())

    housing_data.hist()

    housing_data_with_id = housing_data.reset_index()
    test_size = 0.2

    identifier = "index"
    train_set, test_set = split_train_test(housing_data_with_id, test_size, identifier)
    print("len of train_set = %d, test_set = %d" % (len(train_set), len(test_set)))

    random_state = 42
    stratified_train_set, stratified_test_set = stratified_split_train_test_with_income(housing_data, test_size,
                                                                                        random_state)
    print("len of stratified_train_set = %d, stratified_test_set = %d" % (len(stratified_train_set),
                                                                          len(stratified_test_set)))

    housing_data_copy = housing_data.copy()
    housing_data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    housing_data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                           s=housing_data_copy["population"] / 100, label="population",
                           c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing_data_copy[attributes], figsize=(10, 8))

    # plt.show()

    housing_data_features = housing_data.drop("median_house_value", axis=1)
    housing_data_labels = housing_data["median_house_value"].copy()

    imputer = Imputer(strategy="median")
    housing_data_num_features = housing_data.drop("ocean_proximity", axis=1)
    imputer.fit(housing_data_num_features)

    print_with_header("===== Median of Num Attributes =====", housing_data_num_features.median().values)
    print_with_header("===== Imputer Computed Values  =====", imputer.statistics_)

    X = imputer.transform(housing_data_num_features)
    housing_data_num_tr_features = pd.DataFrame(X, columns=housing_data_num_features.columns)

    print_with_header("===== Info =====", housing_data_num_tr_features.info)

    num_attrs, cat_atts = list(housing_data_num_features), ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attrs)),
        ('imputer', Imputer(strategy='median')),
        ('attributes_adder', CombinedAttributesAdder()),
        ('standard_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_atts)),
        ('label_binarizer', LabelBinarizerPipelineFriendly())
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

    housing_data_prep = full_pipeline.fit_transform(housing_data)
    print_with_header("Housing Data Preprocessed with Pipelines", housing_data_prep)
    print_with_header("Shape", housing_data_prep.shape)

    lin_regression = LinearRegression()
    lin_regression.fit(housing_data_prep, housing_data_labels)

    some_data, some_labels = housing_data.iloc[:5], housing_data_labels.iloc[:5]
    some_data_prep = full_pipeline.transform(some_data)
    print_with_header("Predictions", lin_regression.predict(some_data_prep))
    print_with_header("Labels", some_labels)
