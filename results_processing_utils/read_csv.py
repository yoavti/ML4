import pandas as pd

from functools import partial


def read_indexed_csv(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)
    return df


def parse_arr(s):
    s = s[1:-1]
    return s.split()


def process_scores(s):
    scores = parse_arr(s)
    return ','.join(scores)


def process_features(s):
    features = parse_arr(s)
    features = [feature[1:-1] for feature in features]
    return ','.join(features)


def read_fs(path):
    df = read_indexed_csv(path)
    df['scores'] = df['scores'].apply(process_scores)
    df['features'] = df['features'].apply(process_features)
    return df


def extract_function_name(func):
    return func[10:-19]


def until(s, c):
    idx = s.find(c)
    s = s[:idx]
    return s


def stringify_score_func(s):
    while all(substr in s for substr in ['<function ', ' at 0x', '>']):
        start_idx = s.find('<function ')
        end_idx = s.find('>')
        func = s[start_idx:end_idx+1]
        func_name = extract_function_name(func)
        s = s.replace(func, func_name)
    return s


def read_cv_results(path, verbose_transformer=False):
    if verbose_transformer:
        transformer_parser = stringify_score_func
    else:
        transformer_parser = partial(until, c='(')
    df = read_indexed_csv(path)

    df['param_fs__transformer__score_func'] = df['param_fs__transformer__score_func'].astype(str)
    df['param_fs__transformer__score_func'] = df['param_fs__transformer__score_func'].apply(extract_function_name)

    df['param_fs__transformer'] = df['param_fs__transformer'].astype(str)
    df['param_fs__transformer'] = df['param_fs__transformer'].apply(transformer_parser)

    df['param_clf__estimator'] = df['param_clf__estimator'].astype(str)
    df['param_clf__estimator'] = df['param_clf__estimator'].apply(partial(until, c='()'))
    return df
