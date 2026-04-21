import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

BASE = "/Users/armandsechon/dev/cours/algorithme d'apprentissage/Airbnb/"

train = pd.read_csv(BASE + "airbnb_train.csv")
test = pd.read_csv(BASE + "airbnb_test.csv")

target = train["log_price"]
test_ids = test.iloc[:, 0]

train = train.drop(columns=["log_price"])
all_data = pd.concat([train, test], axis=0, ignore_index=True)


# --- Feature Engineering ---

def count_amenities(s):
    if pd.isna(s) or s == "{}":
        return 0
    return s.count(",") + 1

def has_amenity(s, keyword):
    if pd.isna(s):
        return 0
    return int(keyword.lower() in s.lower())

AMENITY_KEYWORDS = [
    "TV", "Wireless Internet", "Air conditioning", "Kitchen", "Heating",
    "Washer", "Dryer", "Elevator", "Pool", "Gym", "Hot tub",
    "Free parking", "Doorman", "Breakfast", "Laptop friendly",
    "Smoke detector", "Carbon monoxide detector", "Fire extinguisher",
    "Essentials", "Shampoo", "Lock on bedroom door", "Hangers",
    "Hair dryer", "Iron", "Self Check-In",
]

all_data["amenities_count"] = all_data["amenities"].apply(count_amenities)
for kw in AMENITY_KEYWORDS:
    col_name = "has_" + re.sub(r'\W+', '_', kw.lower()).strip('_')
    all_data[col_name] = all_data["amenities"].apply(lambda s, k=kw: has_amenity(s, k))

all_data["cleaning_fee"] = all_data["cleaning_fee"].astype(int)

for col in ["host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
    all_data[col] = (all_data[col] == "t").astype(int)

all_data["host_response_rate"] = (
    all_data["host_response_rate"]
    .str.replace("%", "", regex=False)
    .astype(float)
)
all_data["host_response_rate"] = all_data["host_response_rate"].fillna(all_data["host_response_rate"].median())

for date_col in ["first_review", "last_review", "host_since"]:
    all_data[date_col] = pd.to_datetime(all_data[date_col], errors="coerce")
    all_data[date_col + "_year"] = all_data[date_col].dt.year.fillna(-1)
    all_data[date_col + "_month"] = all_data[date_col].dt.month.fillna(-1)
    ref = pd.Timestamp("2018-01-01")
    all_data[date_col + "_days_since"] = (ref - all_data[date_col]).dt.days.fillna(-1)

all_data["review_span_days"] = (
    all_data["last_review_days_since"] - all_data["first_review_days_since"]
)
all_data["host_tenure_days"] = all_data["host_since_days_since"]

all_data["description_len"] = all_data["description"].fillna("").str.len()
all_data["name_len"] = all_data["name"].fillna("").str.len()
all_data["description_word_count"] = all_data["description"].fillna("").str.split().str.len()

all_data["bathrooms"] = all_data["bathrooms"].fillna(all_data["bathrooms"].median())
all_data["bedrooms"] = all_data["bedrooms"].fillna(all_data["bedrooms"].median())
all_data["beds"] = all_data["beds"].fillna(all_data["beds"].median())
all_data["review_scores_rating"] = all_data["review_scores_rating"].fillna(all_data["review_scores_rating"].median())

all_data["rooms_per_person"] = all_data["bedrooms"] / all_data["accommodates"].clip(lower=1)
all_data["beds_per_person"] = all_data["beds"] / all_data["accommodates"].clip(lower=1)
all_data["bath_per_person"] = all_data["bathrooms"] / all_data["accommodates"].clip(lower=1)
all_data["beds_per_room"] = all_data["beds"] / all_data["bedrooms"].clip(lower=1)

cat_cols = ["property_type", "room_type", "bed_type", "cancellation_policy", "city"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_data[col + "_enc"] = le.fit_transform(all_data[col].fillna("Unknown"))
    label_encoders[col] = le

all_data["neighbourhood"] = all_data["neighbourhood"].fillna("Unknown")
le_neigh = LabelEncoder()
all_data["neighbourhood_enc"] = le_neigh.fit_transform(all_data["neighbourhood"])

all_data["zipcode"] = all_data["zipcode"].fillna("Unknown")
zip_counts = all_data["zipcode"].value_counts()
all_data["zipcode_clean"] = all_data["zipcode"].where(
    all_data["zipcode"].map(zip_counts) >= 5, "rare"
)
le_zip = LabelEncoder()
all_data["zipcode_enc"] = le_zip.fit_transform(all_data["zipcode_clean"])

city_stats = pd.concat([all_data.iloc[:len(train)], target.rename("log_price")], axis=1)
for grp_col in ["city_enc", "neighbourhood_enc", "zipcode_enc", "room_type_enc", "property_type_enc"]:
    means = city_stats.groupby(grp_col)["log_price"].mean()
    all_data[grp_col + "_target_mean"] = all_data[grp_col].map(means).fillna(target.mean())

drop_cols = [
    "id", "amenities", "description", "name", "neighbourhood", "zipcode",
    "zipcode_clean", "first_review", "last_review", "host_since",
    "property_type", "room_type", "bed_type", "cancellation_policy", "city",
]
all_data.drop(columns=[c for c in drop_cols if c in all_data.columns], inplace=True)

# Sanitize column names for LightGBM
all_data.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in all_data.columns]

# --- Split back ---
X_train = all_data.iloc[:len(train)]
X_test = all_data.iloc[len(train):]

print(f"Features: {X_train.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- Cross-validation ---
params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

model = lgb.LGBMRegressor(**params)
scores = cross_val_score(model, X_train, target, cv=5, scoring="neg_root_mean_squared_error")
print(f"CV RMSE: {-scores.mean():.5f} (+/- {scores.std():.5f})")

# --- Train final model and predict ---
model.fit(X_train, target)
preds = model.predict(X_test)

# --- Save predictions ---
output = pd.DataFrame({"": test_ids, "logpred": preds})
output.to_csv(BASE + "prediction.csv", index=False)
print(f"Predictions saved to prediction.csv ({len(preds)} rows)")

# --- Feature importance ---
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop 20 features:")
print(importances.sort_values(ascending=False).head(20))
