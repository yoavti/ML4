import os

import pandas as pd


def remove_first_char(s):
    s = s[1:]
    return s


def replace_whitespace(s, sep=','):
    values = s.split()
    s = sep.join(values)
    return s


def bracketed_to_comma_separated(s):
    s = remove_first_char(s)
    s = replace_whitespace(s)
    return s


def remove_quotes(s, sep=','):
    values = s.split(sep)
    values = [value[1:-1] for value in values]
    s = sep.join(values)
    return s


def read_fs(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)
    df['scores'] = df['scores'].apply(lambda s: replace_whitespace(remove_first_char(s)))
    df['features'] = df['features'].apply(lambda s: remove_quotes(replace_whitespace(remove_first_char(s))))
    return df


def read_cv_results(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)
    return df


def main():
    path = os.path.join('results', 'alon', '10')
    fs = read_fs(os.path.join(path, 'fs.csv'))
    cv_results = read_cv_results(os.path.join(path, 'cv_results.csv'))
    print(cv_results)


if __name__ == '__main__':
    main()
