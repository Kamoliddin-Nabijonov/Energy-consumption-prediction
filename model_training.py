import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

SEED = 42

def prepare_dataset(dataset):
    df_train, df_test = train_test_split(dataset, test_size=0.1, random_state=SEED)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.energyconsumption.values
    y_test = df_test.energyconsumption.values

    del df_train['energyconsumption']
    del df_test['energyconsumption']

    dv = DictVectorizer(sparse=False)

    train_dicts = df_train.to_dict(orient='records')
    test_dicts = df_test.to_dict(orient='records')

    dv.fit(train_dicts)

    X_train = dv.transform(train_dicts)
    X_test = dv.transform(test_dicts)

    return X_train, X_test, y_train, y_test, dv


def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Linear Regression:")
    print(f"Root Mean Squared Error: {rmse}")
    return model


def train_SVR(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("SVR:")
    print(f"Root Mean Squared Error: {rmse}")
    return model


def train_gradient_boosting_regressor(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Gradient Boosting Regressor:")
    print(f"Root Mean Squared Error: {rmse}")
    return model


def train_random_forest_regressor(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200, max_depth=11, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Random Forest:")
    print(f"Root Mean Squared Error: {rmse}")
    return model


def save_model(asset, file_name):
    output_file = file_name
    with open(output_file, 'wb') as f_out:
        pickle.dump(asset, f_out)


def main():
    dataset = pd.read_csv('./datasets/energy_consumption_processed.csv')
    X_train, X_test, y_train, y_test, vectorizer = prepare_dataset(dataset)
    linear_regression_model = train_linear_regression(X_train, X_test, y_train, y_test)
    svr_model = train_SVR(X_train, X_test, y_train, y_test)
    gradient_boosting_regressor_model = train_gradient_boosting_regressor(X_train, X_test, y_train, y_test)
    random_forest_regressor_model = train_random_forest_regressor(X_train, X_test, y_train, y_test)

    save_model(vectorizer, './models/dict_vectorizer.bin')
    save_model(linear_regression_model, './models/linear_regression_model.bin')
    save_model(svr_model, './models/svr_model.bin')
    save_model(gradient_boosting_regressor_model, './models/gradient_boosting_regressor_model.bin')
    save_model(random_forest_regressor_model, './models/random_forest_regressor_model.bin')


if __name__ == "__main__":
    main()
