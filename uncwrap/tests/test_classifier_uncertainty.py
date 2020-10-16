import pytest
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from uncwrap import UncertaintyWrapperEstimator
from uncwrap.tests.test_utils import load_politics_data, PoliticsClassifierWrapper


@pytest.fixture
def newsgroups_data():
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data, newsgroups_train.target

@pytest.fixture
def politics_data():
    return load_politics_data("./uncwrap/tests/fixtures/bbc-text.csv")


def test_pretrained_classifier(newsgroups_data, politics_data):
    # Given
    news_X, news_y = newsgroups_data
    polit_X_train, polit_X_test, polit_y_train, polit_y_test = politics_data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(news_X)

    #   We build a very simple naive bayes model using the 20 newsgroups data that will become the blackbox model
    blackbox_model = MultinomialNB()
    blackbox_model.fit(tfidf_vectorizer.transform(news_X), news_y)

    #   we create the target classifier
    politics_model = PoliticsClassifierWrapper(blackbox_model)

    # When

    #   and use the uncertainty wrapper to check it
    uncertainty_wrapper = UncertaintyWrapperEstimator(black_box=politics_model)
    uncertainty_wrapper.fit(tfidf_vectorizer.transform(polit_X_train), polit_y_train)

    polit_y_pred_uncert = uncertainty_wrapper.predict(tfidf_vectorizer.transform(polit_X_test))

    # Then
    assert_almost_equal(polit_y_pred_uncert[:, 1].max(), 0.693, decimal=3)
    assert_almost_equal(polit_y_pred_uncert[:, 1].min(), 0.000, decimal=3)
    assert_array_equal(politics_model.predict(tfidf_vectorizer.transform(polit_X_test)), polit_y_pred_uncert[:, 0])


def test_no_pretrained_classifier_and_no_preds(newsgroups_data, politics_data):
    # Given
    news_X, news_y = newsgroups_data
    polit_X_train, polit_X_test, polit_y_train, polit_y_test = politics_data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(news_X)

    #   we create the target classifier
    politics_model = None

    # When

    #   and use the uncertainty wrapper to check it
    uncertainty_wrapper = UncertaintyWrapperEstimator(black_box=None)
    # Then
    with pytest.raises(ValueError, match="No blackbox or predictions was set"):
        uncertainty_wrapper.fit(tfidf_vectorizer.transform(polit_X_train), polit_y_train)


def test_pretrained_classifier_wrong_input_shapes(newsgroups_data, politics_data):
    # Given
    news_X, news_y = newsgroups_data
    polit_X_train, polit_X_test, polit_y_train, polit_y_test = politics_data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(news_X)

    #   we create the target classifier
    politics_model = None

    # When
    X = tfidf_vectorizer.transform(polit_X_train)
    y = news_y
    #   and use the uncertainty wrapper to check it
    uncertainty_wrapper = UncertaintyWrapperEstimator(black_box=None)
    # Then
    with pytest.raises(ValueError, match=r'Found input variables with inconsistent numbers of samples: [[\d, \d]'):
        uncertainty_wrapper.fit(X, y)


def test_pretrained_classifier_with_preds(newsgroups_data, politics_data):
    # Given
    news_X, news_y = newsgroups_data
    polit_X_train, polit_X_test, polit_y_train, polit_y_test = politics_data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(news_X)

    #   We build a very simple naive bayes model using the 20 newsgroups data that will become the blackbox model
    blackbox_model = MultinomialNB()
    blackbox_model.fit(tfidf_vectorizer.transform(news_X), news_y)

    #   we create the target classifier
    politics_model = PoliticsClassifierWrapper(blackbox_model)

    # When
    X = tfidf_vectorizer.transform(polit_X_train)
    y = polit_y_train
    pred_y = politics_model.predict(X)

    #   and use the uncertainty wrapper to check it
    uncertainty_wrapper = UncertaintyWrapperEstimator()
    uncertainty_wrapper.fit(X, y, pred_y)

    X_test = tfidf_vectorizer.transform(polit_X_test)
    pred_y_test = politics_model.predict(X_test)
    polit_y_pred_uncert = uncertainty_wrapper.predict(X_test, pred_y_test)

    # Then
    assert_almost_equal(polit_y_pred_uncert[:, 1].max(), 0.693, decimal=3)
    assert_almost_equal(polit_y_pred_uncert[:, 1].min(), 0.000, decimal=3)
    assert_array_equal(pred_y_test, polit_y_pred_uncert[:, 0])