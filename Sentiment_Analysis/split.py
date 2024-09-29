from config import INTERIM_DATA_DIR, PARAMS_DIR
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

df = pd.read_csv(INTERIM_DATA_DIR / "clean_dataset.csv")

with open(PARAMS_DIR, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["split"]
    except yaml.YAMLError as exc:
        print(exc)

y = df["positive"]
X = df.drop(columns="positive")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    train_size=params["train_size"],
    random_state=params["random_state"],
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=params["val_size"],
    train_size=1 - params["val_size"],
    random_state=params["random_state"],
)

X_train.to_csv(str(INTERIM_DATA_DIR) + "/" + "X_train.csv", index=False)
y_train.to_csv(str(INTERIM_DATA_DIR) + "/" + "y_train.csv", index=False)
X_test.to_csv(str(INTERIM_DATA_DIR) + "/" + "X_test.csv", index=False)
y_test.to_csv(str(INTERIM_DATA_DIR) + "/" + "y_test.csv", index=False)
X_val.to_csv(str(INTERIM_DATA_DIR) + "/" + "X_val.csv", index=False)
y_val.to_csv(str(INTERIM_DATA_DIR) + "/" + "y_val.csv", index=False)
