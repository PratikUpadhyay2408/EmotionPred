import os
import re
import pandas as pd

# dict of keys!
VALUE_EXTRACT_KEYS = {
    "age": {
        'search_key': 'Age',
        'delimiter': ':'
    },
    "height": {
        'search_key': 'Height',
        'delimiter': ':'
    },
    "weight": {
        'search_key': 'Weight',
        'delimiter': ':'
    },
    "gender": {
        'search_key': 'Gender',
        'delimiter': ':'
    },
    "dominant_hand": {
        'search_key': 'Dominant',
        'delimiter': ':'
    },
    "coffee_today": {
        'search_key': 'Did you drink coffee today',
        'delimiter': '? '
    },
    "coffee_last_hour": {
        'search_key': 'Did you drink coffee within the last hour',
        'delimiter': '? '
    },
    "sport_today": {
        'search_key': 'Did you do any sports today',
        'delimiter': '? '
    },
    "smoker": {
        'search_key': 'Are you a smoker',
        'delimiter': '? '
    },
    "smoke_last_hour": {
        'search_key': 'Did you smoke within the last hour',
        'delimiter': '? '
    },
    "feel_ill_today": {
        'search_key': 'Do you feel ill today',
        'delimiter': '? '
    }
}

DATA_PATH = 'WESAD/'
parse_file_suffix = '_readme.txt'

dummy_df = ['age', 'height', 'weight', 'gender_ female', 'gender_ male',
            'coffee_today_YES', 'sport_today_YES', 'smoker_NO', 'smoker_YES',
            'feel_ill_today_YES', 'subject']


def parse_readme(file):

    print(f"parsing {file}")

    with open(file, 'r') as f:
        x = f.read().split('\n')

    readme_dict = {}

    for item in x:
        for key in VALUE_EXTRACT_KEYS.keys():
            search_key = VALUE_EXTRACT_KEYS[key]['search_key']
            delimiter = VALUE_EXTRACT_KEYS[key]['delimiter']
            if item.startswith(search_key):
                d, v = item.split(delimiter)
                readme_dict.update({key: v})
                break
    return readme_dict


def parse_all_readmes(sub_list, filenames):

    dframes = []

    for sub,path in zip(sub_list,filenames):
        readme_dict = parse_readme(path)
        print(readme_dict)
        df = pd.DataFrame(readme_dict, index=[sub])
        dframes.append(df)

    df = pd.concat(dframes)
    return  df

if __name__ == "__main__":

    sub_list = []
    filenames = []
    for i in range(1, 18):
        file = f"WESAD/S{i}/S{i}_readme.txt"
        if not os.path.isfile( file ):
            pass
        else:
            filenames.append(file)
            sub_list.append(i)

    info_df = parse_all_readmes(sub_list, filenames)
    info_df.to_csv("MergedData/Subject_Info.csv")