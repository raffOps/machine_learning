import pickle

import pytest
import numpy as np

from src.Kfold.kfold import Kfold


@pytest.fixture
def kfold_k_equal_3():
    return Kfold(k=3)


@pytest.fixture
def kfold_k_equal_5():
    return Kfold(k=5)


@pytest.fixture
def kfold_k_equal_8():
    return Kfold(k=8)


@pytest.fixture
def values():
    with open("mocks/test_values", "rb") as file:
        values = pickle.load(file)
    return values


@pytest.fixture
def values_splitted_in_3_folds():
    with open("mocks/values_splitted_in_3_folds", "rb") as file:
        values = pickle.load(file)
    return values


@pytest.fixture
def values_splitted_in_5_folds():
    with open("mocks/values_splitted_in_5_folds", "rb") as file:
        values = pickle.load(file)
    return values


@pytest.fixture
def values_splitted_in_8_folds():
    with open("mocks/values_splitted_in_8_folds", "rb") as file:
        values = pickle.load(file)
    return values


def test_generate_split_with_k_equal_3(kfold_k_equal_3):
    splits = kfold_k_equal_3.generate_splits()
    assert set(splits) == {
        ((1, 2), 0),
        ((0, 2), 1),
        ((0, 1), 2)
    }


def test_generate_split_with_k_equal_5(kfold_k_equal_5):
    splits = kfold_k_equal_5.generate_splits()
    assert set(splits) == {
        ((0, 1, 3, 4), 2),
        ((0, 2, 3, 4), 1),
        ((0, 1, 2, 4), 3),
        ((1, 2, 3, 4), 0),
        ((0, 1, 2, 3), 4)
    }


def test_generate_split_with_k_equal_8(kfold_k_equal_8):
    splits = kfold_k_equal_8.generate_splits()
    assert set(splits) == {
        ((0, 1, 2, 3, 4, 6, 7), 5),
        ((0, 1, 2, 3, 4, 5, 6), 7),
        ((0, 1, 2, 4, 5, 6, 7), 3),
        ((1, 2, 3, 4, 5, 6, 7), 0),
        ((0, 1, 2, 3, 5, 6, 7), 4),
        ((0, 2, 3, 4, 5, 6, 7), 1),
        ((0, 1, 3, 4, 5, 6, 7), 2),
        ((0, 1, 2, 3, 4, 5, 7), 6)
    }


def test_generate_folds_with_k_equal_3(kfold_k_equal_3, values, values_splitted_in_3_folds):
    folds = kfold_k_equal_3.generate_folds(values)
    assert folds == values_splitted_in_3_folds


def test_generate_folds_with_k_equal_3(kfold_k_equal_3, values, values_splitted_in_3_folds):
    folds = kfold_k_equal_3.generate_folds(values)
    assert folds == values_splitted_in_3_folds


def test_generate_folds_with_k_equal_5(kfold_k_equal_5, values, values_splitted_in_5_folds):
    folds = kfold_k_equal_5.generate_folds(values)
    assert folds == values_splitted_in_5_folds
    
    
def test_generate_folds_with_k_equal_8(kfold_k_equal_8, values, values_splitted_in_8_folds):
    folds = kfold_k_equal_8.generate_folds(values)
    assert folds == values_splitted_in_8_folds


def test_totally_disjointed_3_folds(values_splitted_in_3_folds):
    joined_folds = []
    for fold in values_splitted_in_3_folds:
        joined_folds.extend(fold)

    assert len(joined_folds) == len(set(joined_folds))
    
    
def test_totally_disjointed_5_folds(values_splitted_in_5_folds):
    joined_folds = []
    for fold in values_splitted_in_5_folds:
        joined_folds.extend(fold)

    assert len(joined_folds) == len(set(joined_folds))
    
    
def test_totally_disjointed_8_folds(values_splitted_in_8_folds):
    joined_folds = []
    for fold in values_splitted_in_8_folds:
        joined_folds.extend(fold)

    assert len(joined_folds) == len(set(joined_folds))


def test_balanced_3_folds(values_splitted_in_3_folds, values):
    for fold in values_splitted_in_3_folds:
        assert np.asarray([row[1] for index, row
                           in enumerate(values)
                           if index in fold]
                          ).mean() \
               == 0.5
     
        
def test_balanced_5_folds(values_splitted_in_5_folds, values):
    for fold in values_splitted_in_5_folds:
        assert np.asarray([row[1] for index, row
                           in enumerate(values)
                           if index in fold]
                          ).mean() \
               == 0.5
  
        
def test_balanced_8_folds(values_splitted_in_8_folds, values):
    for fold in values_splitted_in_8_folds:
        assert np.asarray([row[1] for index, row
                           in enumerate(values)
                           if index in fold]
                          ).mean() \
               == 0.5
