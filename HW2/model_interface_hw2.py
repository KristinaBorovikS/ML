import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    from sklearn.preprocessing import StandardScaler
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df),columns=df.columns)
    X = df_scaled.drop(columns=[target_column])
    y = df_scaled[target_column]

    numeric_features = ['Number_of_Customers_Per_Day', 'Average_Order_Value', 'Operating_Hours_Per_Day','Number_of_Employees','Marketing_Spend_Per_Day','Location_Foot_Traffic']
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer( transformers=[ ('num', numeric_transformer, numeric_features)])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def get_result(df,target):
    X, y, preprocessor = preprocess_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    mse, r2 = evaluate_model(y_test, y_pred)
    return mse, r2, y_pred,y_test
