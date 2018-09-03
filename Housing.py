import os
import pandas as pd
import tarfile
import matplotlib.pyplot as plt
import hashlib
import numpy as np

from six.moves import urllib

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


if '__main__' == __name__:
    set_print_options()
    housing_data = load_housing_data()
    print(housing_data['ocean_proximity'].value_counts())
    print_with_header("===== Peek =====", housing_data.head())
    print_with_header("===== Info =====", housing_data.info)
    print_with_header("=== Describe ===", housing_data.describe())
    print_with_header("===== Corr =====", housing_data.corr())
    print_with_header("===== Skew =====", housing_data.skew())

    # housing_data.hist()
    # plt.show()

    housing_data_with_id = housing_data.reset_index()
    test_size, identifier = 0.2, "index"
    train_set, test_set = split_train_test(housing_data_with_id, test_size, identifier)
    print("len of train_set = %d, test_set = %d" % (len(train_set), len(test_set)))
