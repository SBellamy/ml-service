import pandas as pd

REQUIRED_COLUMNS = [
    'age',
    'income',
    'account_balance',
    'transactions_last_30d',
    'is_premium',
    'target'
]

def validate_df(df: pd.DataFrame) -> None:
    # check for required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    
    # check for null values
    if df.isnull().any().any():
        raise ValueError('Null values detected')
    
    # check that target is binary
    if not df["target"].isin([0, 1]).all():
        raise ValueError('Target column must be binary (0 or 1)')
    