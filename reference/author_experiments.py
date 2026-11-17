# CELL 0
import sys
import os
sys.path.append(os.path.abspath(".."))
import matplotlib.pyplot as plt
import pandas as pd
from tests.datasets import load_dataset
from core.model_scores import generate_forecasts
from core.methods import KrichevskyTrofimov, UniversalPortfolio, DynamicallyTunedAdaptiveConformalInference
from core.runner import run_test

# CELL 1
data_name = "elec2"
data = load_dataset(data_name)
data

# CELL 2
model_name = "ar"
overwrite = False
log = False
fit_every = 1
ahead = 1
save_name = "../datasets/processed/" + data_name + "/" + model_name + "-" + str(fit_every) + "-steps" + ".npz"
forecasts = generate_forecasts(data, model_name, save_name, overwrite, log, fit_every, ahead)

# CELL 3
if 'forecasts' not in data.columns:
    data['forecasts'] = forecasts
if 'scores' not in data.columns:
    data['scores'] = abs(data['y'] - data['forecasts'])

# CELL 4
data['scores'].values

# CELL 5
KT = KrichevskyTrofimov(alpha=0.05)
UP = UniversalPortfolio(alpha=0.05)
DTACI = DynamicallyTunedAdaptiveConformalInference(alpha=0.05)
KT_result = run_test(data['scores'].values, KT) 
UP_result = run_test(data['scores'].values, UP)
DTACI_result = run_test(data['scores'].values, DTACI)

# CELL 6
KT_coverages = pd.Series(KT_result['coverages']).rolling(window=100, min_periods=1).mean().to_numpy()
UP_coverages = pd.Series(UP_result['coverages']).rolling(window=100, min_periods=1).mean().to_numpy()
DTACI_coverages = pd.Series(DTACI_result['coverages']).rolling(window=100, min_periods=1).mean().to_numpy()

# CELL 7
plt.plot(KT_coverages, label='KT')
plt.plot(UP_coverages, label='UP')
plt.plot(DTACI_coverages, label='DTACI')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.legend()

# CELL 8
for i in range(len(KT_coverages)):
    print(f"{i}\t{KT_coverages[i]:.4f}\t{UP_coverages[i]:.4f}\t{DTACI_coverages[i]:.4f}")

# CELL 9
