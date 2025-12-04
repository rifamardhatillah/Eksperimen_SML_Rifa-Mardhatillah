import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(input_path: str,
                    output_path: str = "diabetes_preprocessing.csv"):
    # 1. Load dataset
    df = pd.read_csv(input_path)
    print("Shape awal:", df.shape)

    # 2. Tangani nilai 0 yang dianggap missing
    cols_missing_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_missing_zero:
        median_value = df[col].median()
        df[col] = df[col].replace(0, median_value)

    # 3. Hapus duplikat
    df = df.drop_duplicates()
    print("Setelah drop duplicates:", df.shape)

    # 4. Hapus outlier (IQR)
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Outcome")

    for col in numeric_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]

    print("Setelah outlier removal:", df_out.shape)

    # 5. Normalisasi fitur
    X = df_out.drop("Outcome", axis=1)
    y = df_out["Outcome"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 6. Split train–test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # 7. Simpan (opsional, untuk dokumentasi)
    processed_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    processed_df.to_csv(output_path, index=False)
    print(f"Dataset hasil preprocessing disimpan ke: {output_path}")

    # fungsi mengembalikan data siap latih
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Contoh pemanggilan
    X_train, X_test, y_train, y_test = preprocess_data("diabetes_raw.csv")
