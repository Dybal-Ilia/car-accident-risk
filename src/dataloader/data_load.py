import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd


def download_dataset(competition:str, dest_path:str, unzip:bool=True, force:bool=False, quiet:bool=False):

    """Download a provided competition dataset from Kaggle.
    
    Args:
        competition (str): The name of the competition to download the dataset from.
        dest_path (str): The destination path where the dataset will be saved.
        unzip (bool, optional): Whether to unzip the downloaded dataset. Defaults to True.
        force (bool, optional): Whether to force re-download if the dataset already exists. Defaults to False.
        quiet (bool, optional): Whether to suppress output messages. Defaults to False.
    """

    os.makedirs(dest_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(competition, path=dest_path, force=force, quiet=quiet)

    if unzip:
        for fname in os.listdir(dest_path):
            if fname.endswith(".zip"):
                full = os.path.join(dest_path, fname)
                with zipfile.ZipFile(full, "r") as z:
                    z.extractall(dest_path)
                os.remove(full)


def map_data(basepath:str):
    """"Maps file names (without extensions) to their full paths in a given directory.
    Args:
        basepath (str): The base directory path containing the files."""
    mapping = {}
    for file in os.listdir(basepath):
        mapping[file.split('.')[0]] = pd.read_csv(os.path.join(basepath, file))
    return mapping