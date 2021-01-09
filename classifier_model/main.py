import pandas as pd
import numpy as np
from classifier_model import preprocessing as pp
from classifier_model import classifier_model as cm


data = pd.read_csv('C:/MachineLearning/data/docter_test.csv', encoding='CP949')

def main():
     df = pp.drop_null(data, ['총점'])
     df['합격여부'] = df['합격여부'].apply(lambda x :  1 if x == "합격" else 0)
     df = pp.group_by(df, '일련번호')
     df = pp.drop_feature(df, ['연도', '회차'])
     df = pp.get_target(df, ['합격여부'])
     target = df['target']
     df = df['df']
     print(df.head(20))
     df = pp.scaler(df)
     target = target.to_numpy()
     target = target.reshape(-1, )

     train_x, test_x, train_y, test_y = pp.split_data(df, target)
     model = cm.process(train_x, test_x, train_y, test_y)
     #predict = "합격" if cm.predict(model, predict_x) == 1 else "불합격"

     return model
