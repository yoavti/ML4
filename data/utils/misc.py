def split_X_y(df, column):
    y = df[column]
    X = df.drop(column, axis=1)
    return X, y
