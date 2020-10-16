import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from uncwrap import UncertaintyWrapperEstimator
from uncwrap.tests.test_utils import PoliticsClassifierWrapper, load_politics_data


def main():
    """

    This script shows how to apply the uncertainty wrapper to enrich the predictions of a trained classifier with
    the corresponding uncertainty when applied to a new domain of application. The goal is to show how to use
    a classifier already trained for one task in another task, and how we can measure that the blackbox adapts for the
    new task. We will see how wecan identify the cases where the blackbox is less confident on its predictions,
    and how we can increase the accuracy of the resulting system by rejecting those examples that more uncertain.

    It first trains a naive bayes classifier over newsgroups that will be the blackbox in this example.
    Then, it uses it for predicting whether a new text is talking about politics or not, by grouping all the
    newsgroups that are related to politics.
    As the target application, it uses the dataset on BBC news(https://www.kaggle.com/yufengdev/bbc-fulltext-and-category),
    making a binary classifier to predict if a news talks about politics or not.
    """

    # First we load the 20 newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    X_train = newsgroups_train.data
    X_test = newsgroups_test.data

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    # We create a tf/idf vectorizer for texts. It is important to remark here that it will be the same that we use
    # for computing the uncertainty, as we need to use the same representation for obtaining the predictions when
    # calling the blackbox trained for newsgroups.
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(X_train)

    # We build a very simple naive bayes model using the 20 newsgroups data that will become the blackbox model
    blackbox_model = MultinomialNB()
    blackbox_model.fit(tfidf_vectorizer.transform(X_train), y_train)

    # We compute the accuracy of the 20 newsgroups dataset to check that the blackbox works
    y_pred = blackbox_model.predict(tfidf_vectorizer.transform(X_test))
    print('accuracy of the black-box in test: {}'.format(accuracy_score(y_test, y_pred)))

    # To apply the blackbox for the binary problem of predicting if a news talks about politics or not
    # we create a wrapper that returns 1 for those news that are related to politics and 0 otherwise
    politics_model = PoliticsClassifierWrapper(blackbox_model)
    y_pred2 = politics_model.predict(tfidf_vectorizer.transform(X_test))

    # Load the BBC news dataset also making a transformation for getting 1 when the category is politics
    # and 0 otherwise
    polit_X_train, polit_X_test, polit_y_train, polit_y_test = load_politics_data("..//tests/fixtures/bbc-text.csv")

    # we create the uncertaintywrapper referencing to the black box
    uncertainty_wrapper = UncertaintyWrapperEstimator(black_box=politics_model, verbose=1)

    # and train the wrapper (mind that we are using the same vectorizer for the input as in the blackbox)
    uncertainty_wrapper.fit(tfidf_vectorizer.transform(polit_X_train), polit_y_train)

    # then we can obtain pairs of predictions and uncertainties
    polit_y_pred_uncert = uncertainty_wrapper.predict(tfidf_vectorizer.transform(polit_X_test))

    # and compare the accuracy when removing the predictions that are more uncertain in the new problem
    polit_y_pred = polit_y_pred_uncert[:, 0]
    polit_y_uncerts = polit_y_pred_uncert[:, 1]
    print('accuracy of the black-box for politics: {}'.format(accuracy_score(polit_y_test, polit_y_pred)))
    UNCERTAINTY_THRESHOLD = 0.69
    polit_y_uncertain_preds_indexes = polit_y_uncerts < UNCERTAINTY_THRESHOLD
    print('accuracy of the black-box for politics for keeping the more {} confident preds out of {}: {}'.format(
        np.count_nonzero(polit_y_uncertain_preds_indexes),
        polit_y_uncerts.shape[0],
        accuracy_score(np.array(polit_y_test)[polit_y_uncertain_preds_indexes],
                       polit_y_pred[polit_y_uncertain_preds_indexes])))

if __name__ == '__main__':  # pragma: no cover
    main()