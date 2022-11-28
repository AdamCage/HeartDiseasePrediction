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