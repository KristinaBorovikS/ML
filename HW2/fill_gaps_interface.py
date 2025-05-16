def fill_gaps_median(df,x):
    df[x.name] = df[x.name].fillna(df[x.name].median())
    print(f"Данные с медианными значениями:")
    print(df.head())

def fill_gaps_mean(df,x):
    df[x.name] = df[x.name].fillna(df[x.name].mean())
    print(f"Данные со средними значениями:")
    print(df.head())

def fill_gaps_mode(df,x):
    df[x.name] = df[x.name].fillna(df[x.name].mode().iloc[0])
    print(f"Данные с наиболее часто встречающимися значениями:")
    print(df.head())
