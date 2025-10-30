from __future__ import annotations, absolute_import, division, print_function, unicode_literals


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
Lotto/Lottery Prediction 

Klasicni regresori: 
✅ DecisionTreeRegressor.
✅ KNeighborsRegressor.
✅ RandomForestRegressor.
✅ LinearRegression.
✅ GradientBoostingRegressor.

svih 4502 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 28.10.2025.
"""

print()
print("🔁 Script started ...")
print()
"""
🔁 Script started ...
"""

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)

from qiskit.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="qiskit.qasm3")


import pytz
from datetime import datetime
import time


# Postavljanje vremena
time = datetime.now(pytz.timezone('Europe/Belgrade')).strftime('%d.%m.%Y_%H.%M.%S')
print()
print(f'\nStart {time}.\n')
print()
"""
Start 29.10.2025_21.16.52.
"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from qiskit_machine_learning.utils import algorithm_globals
import random


import xgboost as xgb

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import _regression, mean_absolute_error, mean_squared_log_error, median_absolute_error, explained_variance_score

from sklearn.metrics import accuracy_score





# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


# ✅ Load data
df = pd.read_csv("/data/loto7_4502_k85.csv", header=None)
print()
print("✅ Data loaded successfully.")
print()
"""
✅ Data loaded successfully.
"""




print()
print(df.shape)
print()
# (4502, 7)   4502 observations of 7 variables


print()
print(df.describe().transpose())
print()
# transposed summary statistics of the variables
"""
    count       mean       std   min   25%   50%   75%   max
0  4502.0   5.151710  4.032235   1.0   2.0   4.0   7.0  28.0
1  4502.0  10.033541  5.200019   2.0   6.0   9.0  13.0  32.0
2  4502.0  15.005109  5.843073   3.0  11.0  15.0  19.0  34.0
3  4502.0  20.057530  6.021846   5.0  16.0  20.0  24.0  36.0
4  4502.0  25.055087  5.762236   6.0  21.0  25.0  29.0  37.0
5  4502.0  30.117503  5.139832   9.0  27.0  31.0  34.0  38.0
6  4502.0  35.056641  3.929804  12.0  33.0  36.0  38.0  39.0
"""






len_data = df.shape[:1][0]
print()
print("len_data = df.shape[:1][0]")
print(len_data)
print()
"""
len_data = df.shape[:1][0]
4502
"""



###################################


print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

      0   1   2   3   4   5   6
4497  4  13  14  19  27  35  37
4498  1   7  13  18  25  30  34
4499  1   5   6   7  11  24  37
4500  2   4   6  11  21  33  35
4501  1   3  11  12  19  35  38
"""


# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)



####################################



# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)



print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

    0   1   2   3   4   5   6
0   4  12  12  13  23  24  27
1   1   1  10  14  14  17  30
2  12  15  15  16  16  20  32
3  16  18  20  22  30  30  31
4   2   2   5   7  24  26  30
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:

      0   1   2   3   4   5   6
4497  3  11  11  15  22  29  30
4498  0   5  10  14  20  24  27
4499  0   3   3   3   6  18  30
4500  1   2   3   7  16  27  28
4501  0   1   8   8  14  29  31
"""

# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df_indexed = df_indexed.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values


# ✅ Train-test split mapiranih brojeva u indeksiranom opsegu
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(X_x, y_x, test_size=0.25, random_state=39)



########################################

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_x = scaler.fit_transform(X_train_x)
X_test_scaled_x = scaler.transform(X_test_x)





# Train a Decision Tree
reg = DecisionTreeRegressor(
    random_state=39,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=20)

reg.fit(X_train_scaled, y_train)


reg_x = DecisionTreeRegressor(
    random_state=39,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=20)

reg_x.fit(X_train_scaled_x, y_train_x)






param_grid = {
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 12, 14]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=39),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_reg = grid_search.best_estimator_

print()
print("Best Parameters:", best_params)
print("Best Estimator:", best_reg)
print()
"""
Best Parameters: {'min_samples_leaf': 1, 'min_samples_split': 2}
Best Estimator: DecisionTreeRegressor(random_state=39)
"""



########################################


# 🔁 Train model
dtr_model = DecisionTreeRegressor(criterion="squared_error", splitter="best", random_state=SEED, 
                                  min_samples_split=2, min_samples_leaf=1, max_depth=20)
KNN_model = KNeighborsRegressor(n_neighbors=5)
rfr_model = RandomForestRegressor(criterion="squared_error", n_estimators=500, max_depth=5, random_state=SEED)
lr_model  = LinearRegression()
gbr_model = MultiOutputRegressor(
    GradientBoostingRegressor(criterion="squared_error", n_estimators=500, max_depth=5, random_state=SEED)
)

print("\n⚛️ Počinje treniranje modela ...")
dtr_model.fit(X_train_scaled, y_train)
KNN_model.fit(X_train,y_train)
rfr_model.fit(X_train,y_train)
lr_model.fit(X_train,y_train)
gbr_model.fit(X_train,y_train)
print("✅ DecisionTreeRegressor trained.")
print("✅ KNeighborsRegressor trained.")
print("✅ RandomForestRegressor trained.")
print("✅ LinearRegression trained.")
print("✅ GradientBoostingRegressor trained.")
print()
"""
⚛️ Počinje treniranje modela ...
✅ DecisionTreeRegressor trained.
✅ KNeighborsRegressor trained.
✅ RandomForestRegressor trained.
✅ LinearRegression trained.
✅ GradientBoostingRegressor trained.
"""


# 🎲 Predict and evaluate
predicted_numbers_dtr = dtr_model.predict(X_test[0].reshape(1, -1))
predicted_numbers_KNN = KNN_model.predict(X_test[0].reshape(1, -1))
predicted_numbers_rfr = rfr_model.predict(X_test[0].reshape(1, -1))
predicted_numbers_lr = lr_model.predict(X_test[0].reshape(1, -1))
predicted_numbers_gbr = gbr_model.predict(X_test[0].reshape(1, -1))



# Convert predictions to integers   🟢 zaokruzimo
predicted_numbers_dtr = np.round(predicted_numbers_dtr).astype(int)
predicted_numbers_KNN = np.round(predicted_numbers_KNN).astype(int)
predicted_numbers_rfr = np.round(predicted_numbers_rfr).astype(int)
predicted_numbers_lr = np.round(predicted_numbers_lr).astype(int)
predicted_numbers_gbr = np.round(predicted_numbers_gbr).astype(int)


print()
print("🔮 Predicted Next Lottery Numbers dtr X y:", predicted_numbers_dtr)
print("🔮 Predicted Next Lottery Numbers KNN X y:", predicted_numbers_KNN)
print("🔮 Predicted Next Lottery Numbers rfr X y:", predicted_numbers_rfr)
print("🔮 Predicted Next Lottery Numbers lr X y:", predicted_numbers_lr)
print("🔮 Predicted Next Lottery Numbers gbr X y:", predicted_numbers_gbr)
print()
"""
🔮 Predicted Next Lottery Numbers dtr X y: [[ 3  5 x x x 37 39]]
🔮 Predicted Next Lottery Numbers KNN X y: [[ 5  9 x x x 32 35]]
🔮 Predicted Next Lottery Numbers rfr X y: [[ 5 10 x x x 30 35]]
🔮 Predicted Next Lottery Numbers lr X y: [[ 5 10 15 20 25 30 35]]
🔮 Predicted Next Lottery Numbers gbr X y: [[ 5 12 x x x 31 37]]
"""


#######################################


# 🔁 Train model for mapped data 
dtr_model_x = DecisionTreeRegressor(criterion="squared_error", splitter="best", random_state=SEED, 
                                    min_samples_split=2, min_samples_leaf=1, max_depth=20)
KNN_model_x = KNeighborsRegressor(n_neighbors=5)
rfr_model_x = RandomForestRegressor(criterion="squared_error", n_estimators=500, max_depth=5, random_state=SEED)
lr_model_x  = LinearRegression()
gbr_model_x = MultiOutputRegressor(
    GradientBoostingRegressor(criterion="squared_error", n_estimators=500, max_depth=5, random_state=SEED)
)


print("\n⚛️ Počinje treniranje modela ...")
dtr_model_x.fit(X_train_scaled_x, y_train_x)
KNN_model_x.fit(X_train_x, y_train_x)
rfr_model_x.fit(X_train_x, y_train_x)
lr_model_x.fit(X_train_x, y_train_x)
gbr_model_x.fit(X_train_x, y_train_x)
print("✅ DecisionTreeRegressor trained for mapped data.")
print("✅ KNeighborsRegressor trained for mapped data.")
print("✅ RandomForestRegressor trained for mapped data.")
print("✅ LinearRegression trained for mapped data.")
print("✅ GradientBoostingRegressor trained for mapped data.")
print()
"""
⚛️ Počinje treniranje modela ...
✅ DecisionTreeRegressor trained for mapped data.
✅ KNeighborsRegressor trained for mapped data.
✅ RandomForestRegressor trained for mapped data.
✅ LinearRegression trained for mapped data.
✅ GradientBoostingRegressor trained for mapped data.
"""


# 🎲 Predict and evaluate for mapped data
predicted_numbers_dtr_x = dtr_model_x.predict(X_test_x[0].reshape(1, -1))
predicted_numbers_KNN_x = KNN_model_x.predict(X_test_x[0].reshape(1, -1))
predicted_numbers_rfr_x = rfr_model_x.predict(X_test_x[0].reshape(1, -1))
predicted_numbers_lr_x = lr_model_x.predict(X_test_x[0].reshape(1, -1))
predicted_numbers_gbr_x = gbr_model_x.predict(X_test_x[0].reshape(1, -1))



# Convert predictions to integers   🟢 zaokruzimo
predicted_numbers_dtr_x = np.round(predicted_numbers_dtr_x).astype(int)
predicted_numbers_KNN_x = np.round(predicted_numbers_KNN_x).astype(int)
predicted_numbers_rfr_x = np.round(predicted_numbers_rfr_x).astype(int)
predicted_numbers_lr_x = np.round(predicted_numbers_lr_x).astype(int)
predicted_numbers_gbr_x = np.round(predicted_numbers_gbr_x).astype(int)



print()
print("🎯 Predicted Next Lottery Numbers dtr X_x y_x for mapped data:", predicted_numbers_dtr_x)
print("🎯 Predicted Next Lottery Numbers KNN X_x y_x for mapped data:", predicted_numbers_KNN_x)
print("🎯 Predicted Next Lottery Numbers rfr X_x y_x for mapped data:",predicted_numbers_rfr_x)
print("🎯 Predicted Next Lottery Numbers lr X_x y_x for mapped data:", predicted_numbers_lr_x)
print("🎯 Predicted Next Lottery Numbers gbr X_x y_x for mapped data:",predicted_numbers_gbr_x)

print()
"""
🎯 Predicted Next Lottery Numbers dtr X_x y_x for mapped data: [[ 2  3 x x x 31 32]]
🎯 Predicted Next Lottery Numbers KNN X_x y_x for mapped data: [[ 4  7 x x x 26 28]]
🎯 Predicted Next Lottery Numbers rfr X_x y_x for mapped data: [[ 4  8 x x x 24 28]]
🎯 Predicted Next Lottery Numbers lr X_x y_x for mapped data: [[ 4  8 x x x 24 28]]
🎯 Predicted Next Lottery Numbers gbr X_x y_x for mapped data: [[ 4 10 x x x 25 30]]
"""


#######################################


# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4502, Broj pozicija: 7
"""

time = datetime.now(pytz.timezone('Europe/Belgrade')).strftime('%d.%m.%Y_%H.%M.%S')
print()
print(f'\nStop {time}.\n')
print()
"""
Stop 29.10.2025_21.17.17.
"""

#######################################


print("\n✅ Script finished successfully.\n")
"""
✅ Script finished successfully.
"""



###############  

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print()
print(X_scale)
print()
"""
[[0.14814815 0.4        0.38709677 ... 0.70967742 0.72413793 0.81481481]
 [0.03703704 0.03333333 0.32258065 ... 0.41935484 0.48275862 0.92592593]
 [0.44444444 0.5        0.48387097 ... 0.48387097 0.5862069  1.        ]
 ...
 [0.         0.16666667 0.32258065 ... 0.61290323 0.72413793 0.81481481]
 [0.         0.1        0.09677419 ... 0.16129032 0.51724138 0.92592593]
 [0.03703704 0.06666667 0.09677419 ... 0.48387097 0.82758621 0.85185185]]
"""




X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scale, y, test_size=0.25, random_state=39)

X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

# we now have a total of six variables 

print()
print(print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape))
print()
"""
(3375, 7) (563, 7) (563, 7) (3375, 7) (563, 7) (563, 7)
None
"""

