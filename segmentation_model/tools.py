from os.path import isfile, join
from os import listdir
from typing import Text, List, Dict

def read_files_in_folder(folder_path:Text)->List:
        folder_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        folder_files = [join(folder_path, file_path) for file_path in folder_files]
        return folder_files

def append_metrics(dict_store:Dict, current_metrics_dict:Dict):
        for key in current_metrics_dict.keys():
                if key not in dict_store.keys():
                        dict_store[key]=[current_metrics_dict[key]]
                else:
                        dict_store[key].append([current_metrics_dict[key]])

