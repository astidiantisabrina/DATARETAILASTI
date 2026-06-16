import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ====================================
# MEMBACA DATASET
# ====================================
print("Memuat dataset...")

df = pd.read_csv("scanner_data.csv")

# Mengambil 10000 data agar training lebih cepat
df = df.sample(n=10000, random_state=42)

# Menghapus kolom indeks jika ada
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ====================================
# FEATURE ENGINEERING
# ====================================
print("Melakukan feature engineering...")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df.drop(columns=["Date"], inplace=True)

# ====================================
# MENENTUKAN TARGET
# ====================================
target = "SKU_Category"

label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Memisahkan fitur dan target
X = df.drop(columns=[target])
y = df[target]

# ====================================
# KOLOM NUMERIK DAN KATEGORIK
# ====================================
numeric_features = [
    "Customer_ID",
    "Transaction_ID",
    "Quantity",
    "Sales_Amount",
    "Day",
    "Month",
    "Year"
]

categorical_features = [
    "SKU"
]

# ====================================
# PREPROCESSING
# ====================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ====================================
# MEMBAGI DATA
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah data training :", len(X_train))
print("Jumlah data testing  :", len(X_test))

# ====================================
# MEMBUAT MODEL SVC
# ====================================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(
            random_state=42,
            max_iter=10000
        ))
    ]
)

# ====================================
# TRAINING MODEL
# ====================================
print("\nTraining model...")

start = time.time()

model.fit(X_train, y_train)

end = time.time()

print("Training selesai.")
print("Waktu training :", round(end - start, 2), "detik")

# ====================================
# PREDIKSI
# ====================================
y_pred = model.predict(X_test)

# ====================================
# EVALUASI MODEL
# ====================================
accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("HASIL EVALUASI MODEL")
print("================================")

print("Accuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nProgram selesai.")
