import pytest
import pandas as pd

import sys
sys.path.append(".")

from trajectoryProcessing import is_resorted

@pytest.fixture
def sequence():
    return list(range(22))

def test_sort_sequence(sequence):
    data = pd.DataFrame.from_dict({
        'elemento':list('abcdefgh'),
        'ordenInicial':[0,1,2,3,4,5,6,7],
        'ordenFinal':[1,3,2,5,4,7,6,0],}).sort_values('ordenFinal').set_index('elemento')
    acc_list=[set(), set()]
    result = data.apply(is_resorted, args=[acc_list,False], axis=1)

    assert result.to_list() == [True, False, True, False, True, False, True, False]

def test_sort_sequence(sequence):
    data = pd.DataFrame.from_dict({
    "elemento": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
    "ordenInicial": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "ordenFinal": [1, 4, 2, 5, 3, 6, 8, 10, 11, 9, 7, 12, 0],
}).sort_values('ordenFinal').set_index('elemento')
    acc_list=[set(), set()]
    result = data.apply(is_resorted, args=[acc_list,False], axis=1)

    assert result.to_list() == [True, False, True, True, False, False, 
                                False, True, False, True, False, False, False]