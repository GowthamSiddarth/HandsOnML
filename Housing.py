import os
import pandas as pd
import tarfile

from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_CSV = "housing.csv"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


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


housing_data = load_housing_data()
print(housing_data.head())
