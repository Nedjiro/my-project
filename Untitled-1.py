# =========================================
# 1. ИМПОРТ БИБЛИОТЕК
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# =========================================
# 2. ЗАГРУЗКА ДАННЫХ
# =========================================
df = pd.read_csv("train.csv")

print("Размер датасета:", df.shape)
df.head()

# =========================================
# 3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# =========================================
df.describe()

# =========================================
# 4. EDA — ВИЗУАЛИЗАЦИИ
# =========================================

# Распределение цен
plt.figure(figsize=(8,5))
sns.histplot(df["SalePrice"], bins=50, kde=True)
plt.title("Распределение цен на жильё")
plt.show()

# Цена vs площадь
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Площадь vs Цена")
plt.show()

# Цена vs количество комнат
plt.figure(figsize=(8,5))
sns.boxplot(x=df["OverallQual"], y=df["SalePrice"])
plt.title("Качество дома и цена")
plt.show()

# Корреляционная матрица
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.show()

# =========================================
# 5. ОЧИСТКА ДАННЫХ
# =========================================

# удаляем строки без целевой переменной
df = df.dropna(subset=["SalePrice"])

# заполняем пропуски
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("Unknown")

# =========================================
# 6. ПОДГОТОВКА К МОДЕЛИРОВАНИЮ
# =========================================

target = "SalePrice"

features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars",
    "Neighborhood",
    "HouseStyle"
]

X = df[features]
y = df[target]

numeric_features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars"
]

categorical_features = [
    "Neighborhood",
    "HouseStyle"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================================
# 7. TRAIN / TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 8. МОДЕЛИ РЕГРЕССИИ
# =========================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R2": r2_score(y_test, preds)
    })

results_df = pd.DataFrame(results)
results_df

# =========================================
# 9. GRID SEARCH (Random FOREST)
# =========================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# =========================================
# 10. ФИНАЛЬНАЯ ОЦЕНКА
# =========================================

final_preds = best_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, final_preds))
print("RMSE:", mean_squared_error(y_test, final_preds, squared=False))
print("R2:", r2_score(y_test, final_preds))

# =========================================
# 11. ВЫВОДЫ
# =========================================
print("""
Выводы:
1. Площадь и качество дома сильнее всего влияют на цену.
2. Ансамблевые модели показали лучшую точность.
3. Модель подходит для практического прогнозирования цен.
""")
