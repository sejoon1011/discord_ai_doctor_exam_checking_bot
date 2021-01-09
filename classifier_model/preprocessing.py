import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
st = None
def fill_na(df, feature, avg):
    '''
    df: Data you  want to process
    feature: feature that you want to fill
    avg: What you want to fill in df
    What it is: Fill null into the df you put in
    '''
    df[feature].fillna(avg)
    return df
def group_by(df, feature):
    '''
    df: Data you  want to process
    feature: what you want to group by
    '''
    df = df.groupby(feature).mean()
    return df
def drop_feature(df, feature):
    '''
    df: Data you  want to process
    feature: what you want to drop
    '''
    df = df.drop(feature, axis=1)
    return df
def drop_null(df, feature):
    '''
    df: Data you  want to process
    feature: What you want to process
    '''
    df = df.dropna(subset=feature)
    return df
def get_target(df, feature):
    '''
    feature that's target
    '''
    target = df[feature]
    df = df.drop(feature, axis=1)
    return {
        'df' : df,
        'target' : target
    }
def scaler(df):
    global st
    st = StandardScaler()
    st.fit(df)
    df = st.transform(df)
    return df
def st_trans(x_data):
    data = st.transform(x_data)
    return data
def split_data(df, target):
    train_x, test_x, train_y, test_y = train_test_split(df, target, random_state=42, test_size=0.2)

    return [train_x, test_x, train_y, test_y]
