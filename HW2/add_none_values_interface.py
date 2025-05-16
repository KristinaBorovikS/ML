import numpy as np

def add_none_values(df,x):
    for i in range(0, 500):
        df.at[i, x.name] = 0
    df = df.replace([0], [None])
    print(f"Данные с пропущенными значениями:")
    print(df.head())
    print(df.info())
    df = df.fillna(value=np.nan)
    print(f"Наличие значения NaN:")
    print(df.head())
    print(df.isna())
    return df
