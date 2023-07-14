from os.path import isfile, join
from os import listdir
from typing import Text, List

def read_files_in_folder(folder_path:Text)->List:
        """
        Returns all files in folder.
        """
        folder_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        return folder_files
