
import pandas as pd

def load_data(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df['Discount'] = df['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df
