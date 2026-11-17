import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

print(df.columns.to_list())

X = df.drop(columns=["gender","diabetes"])
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numerical_feature = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_feature = X_train.select_dtypes(
    include=["object"]
).columns

numerical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_feature),
    ("cat", categorical_pipeline, categorical_feature)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("lightgbm", lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        verbosity=-1
    ))
])
model.fit(X_train, y_train)
joblib.dump(model,"diabetes_model.pk1")
print("model saved successfully")
