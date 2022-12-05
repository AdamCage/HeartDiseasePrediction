import pandas as pd
import numpy as np

# Config
MODEL_PATH = '.\models\model 9043 05-12-2022 12-05.json'
TEST_MODE = True
print(f'Is test mode: {TEST_MODE}')
print()


def get_oldpeak_cats(row):
    if row < 0:
        return '0-'
    elif row == 0:
        return '[0 - 0]'
    elif row > 0 and row <= 1:
        return '(0 - 1]'
    elif row > 1 and row <= 2:
        return '(1 - 2]'
    elif row > 2 and row <= 3:
        return '(2 - 3]'
    elif row > 3:
        return '3+'


def get_obs_cholestirol(row):
    if row == 0:
        return '[0 - 0]'
    elif row <= 217:
        return '(84.999, 217.0]'
    elif row <= 263:
        return '(217.0, 263.0]'
    else:
        return '(263.0, 603.0]'


def get_obs_age(row):
    if row < 33:
        return '(27.951, 32.9]'
    elif row < 38:
        return '(32.9, 37.8]'
    elif row < 43:
        return '(37.8, 42.7]'
    elif row < 48:
        return '(42.7, 47.6]'
    elif row < 53:
        return '(47.6, 52.5]'
    elif row < 58:
        return '(52.5, 57.4]'
    elif row < 63:
        return '(57.4, 62.3]'
    elif row < 68:
        return '(62.3, 67.2]'
    elif row < 73:
        return '(67.2, 72.1]'
    else:
        return '(72.1, 77.0]'


def get_obs_resting(row):
    if row <= 120:
        return '(-0.001, 120.0]'
    elif row <= 128:
        return '(120.0, 128.0]'
    elif row <= 135.2:
        return '(128.0, 135.2]'
    elif row <= 145:
        return '(135.2, 145.0]'
    else:
        return '(145.0, 200.0]'


def get_obs_hr(row):
    if row <= 103:
        return '(59.999, 103.0]'
    elif row <= 115:
        return '(103.0, 115.0]'
    elif row <= 122:
        return '(115.0, 122.0]'
    elif row <= 130:
        return '(122.0, 130.0]'
    elif row <= 144:
        return '(138.0, 144.0]'
    elif row <= 151:
        return '(144.0, 151.0]'
    elif row <= 160:
        return '(151.0, 160.0]'
    elif row <= 170:
        return '(160.0, 170.0]'
    else:
        return '(170.0, 202.0]'


# Information inputs
print('Input patient information:')

print('Input Gender (M or F)')
gender = str(input()) if not TEST_MODE else ['M', 'F'][np.random.randint(0, 2)]

print('Input Age')
age = int(input()) if not TEST_MODE else np.random.randint(28, 78)

print('Input Chest Pain Type (ASY, ATA, NAP or TA)')
chest_pain = str(input()) if not TEST_MODE else ['ASY', 'ATA', 'NAP', 'TA'][np.random.randint(0, 4)]

print('Input Resting BP')
resting_bp = int(input()) if not TEST_MODE else np.random.randint(0, 201)

print('Input Cholesterol')
cholesterol = int(input()) if not TEST_MODE else np.random.randint(0, 604)

print('Input FastingBS')
fasting_bs = int(input()) if not TEST_MODE else np.random.randint(0, 2)

print('Input RestingECG (ST, Normal, LVH)')
resting_ecg = str(input()) if not TEST_MODE else ['ST', 'Normal', 'LVH'][np.random.randint(0, 3)]

print('Input Max HR')
max_hr = int(input()) if not TEST_MODE else np.random.randint(60, 203)

print('Input ExerciseAngina (Y or N)')
exercise_angina = str(input()) if not TEST_MODE else ['Y', 'N'][np.random.randint(0, 2)]

print('Input Oldpeak')
oldpeak = float(input()) if not TEST_MODE else np.random.randint(-26, 63) / 10

print('Input ST_Slope (Up, Flat, Down)')
st_slope = str(input()) if not TEST_MODE else ['Y', 'N'][np.random.randint(0, 2)]


# UUID and datetime creating
import uuid
id = str(uuid.uuid4())

import datetime
date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M")

# Patient information formalization
observation = pd.DataFrame(
    {
        'id':               [id],
        'date':             [date],
        'Age':              [age],
        'Gender':           [gender],
        'ChestPainType':    [chest_pain],
        'RestingBP':        [resting_bp],
        'Cholesterol':      [cholesterol],
        'FastingBS':        [fasting_bs],
        'RestingECG':       [resting_ecg],
        'MaxHR':            [max_hr],
        'ExerciseAngina':   [exercise_angina],
        'Oldpeak':          [oldpeak],
        'ST_Slope':         [st_slope],
        'ModelPrediction':  [np.nan],
        'PredictionProba':  [np.nan]
    }
)


# Checking and prediction
print('Your patient information:')
print()
print(observation.T)
print()
print('To predict the risk of developing heart disease, enter "predict"')
start = str(input())

if start == 'predict':
    obs = pd.DataFrame(
        {
            'c_age':             get_obs_age(observation['Age'][0]),
            'c_chest_pain':      observation['ChestPainType'],
            'c_resting_bp':      get_obs_resting(observation['RestingBP'][0]),
            'c_cholesterol':     get_obs_cholestirol(observation['Cholesterol'][0]),
            'c_resting_ecg':     observation['RestingECG'],
            'c_max_hr':          get_obs_hr(observation['MaxHR'][0]),
            'c_exercise_angina': observation['ExerciseAngina'],
            'c_oldpeak':         get_oldpeak_cats(observation['Oldpeak'][0]),
            'c_st_slope':        observation['ST_Slope'],

            # Binary features
            'b_gender':          observation['Gender'],
            'b_fasting_bs':      observation['FastingBS'].astype('object'),
        }
    )

# Model loading
from catboost import CatBoostClassifier
model = CatBoostClassifier().load_model(MODEL_PATH, format='json')

# Prediction
pred = model.predict(obs)[0]
pred_proba = f'{round(model.predict_proba(obs)[0][1], 4) * 100} %'

print('Model predictions:', pred)
print('Likelihood of heart disease:', pred_proba)

# Prediction formalization
observation['ModelPrediction'] = pred
observation['PredictionProba'] = pred_proba

observation = observation.T

# Logging into a file
observation.to_excel(f'.\patients\patient {id} {date}.xlsx')

print('Patient information saved')
