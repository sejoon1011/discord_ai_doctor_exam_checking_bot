
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate

def dt_model(train_x, test_x, train_y, test_y):
    model = DecisionTreeClassifier(random_state=42 ,max_depth=5)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    score = accuracy_score(test_y, prediction)
    recall = recall_score(test_y, prediction, average='macro')
    precision = precision_score(test_y, prediction, average='macro')
    return model
def sgb_model(train_x, test_x, train_y, test_y):
    model = SGDClassifier(loss='log', max_iter=100, random_state=42)
    model.fit(train_x, train_y)
    print(model.score(test_x, test_y))
    return model
def rf_model(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return model

def predict(model, prediction_value):
    value = model.predict(prediction_value)
    predict = "합격" if value == 1 else "불합격"
    return predict

def split_data(data, target):
    train_x, test_x, train_y, test_y = train_test_split(data, target, random_state=42)

def process(train_x, test_x, train_y, test_y):
    model = sgb_model(train_x, test_x, train_y, test_y)

    return model

