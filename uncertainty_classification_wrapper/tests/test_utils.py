import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


def load_politics_data(data_path):
    data = pd.read_csv(data_path)

    def process_label(row):
        if row == 'politics':
            return 1
        else:
            return 0

    data['label'] = data['category'].apply(process_label)
    train_data, test_data = train_test_split(data, test_size=0.2)
    return list(train_data['text'].values), list(test_data['text'].values), \
           list(train_data['label'].values), list(test_data['label'].values)


class PoliticsClassifierWrapper(BaseEstimator):
    def __init__(self, black_box: BaseEstimator = None):
        self.black_box = black_box

    def predict(self, X):
        orig_preds = self.black_box.predict(X)
        # we select the cats belonging to politics
        politics_categories = [16, 17, 18, 19]
        y_pred = np.array([int(pred in politics_categories) for pred in orig_preds])
        return y_pred