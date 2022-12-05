import pandas as pd
import numpy as np



def create_general_analysis(df, asc=False):
    unique_values = []
    for i in df.columns:
        unique_values.append(df[i].sort_values(ascending=asc).unique())  # asc

    table_scan = pd.DataFrame(
        {
            'values_num': df.count(),
            'nan_values_num': df.isna().sum(),
            'occupancy': 100 - (df.isna().sum() / (df.isna().sum() + df.notna().sum()) * 100),
            'unique_values_num': df.nunique(),
            # 'min_value': df.min(),
            # 'max_value': df.max(),
            'unique_values': unique_values,
            'dtype': df.dtypes
        }
    )
    print('General data analysis:')
    print()
    print('Shape of the table:     ', df.shape)
    print(f'Duplicates in the table: {df.duplicated().sum()}, ({round(df.duplicated().sum() / df.shape[0], 4) * 100}%)')

    return table_scan


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