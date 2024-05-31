from pytractions.base import (
    TList,
    Out,
    In,
)

from pytractions.transformations import Flatten, FilterDuplicates, ListMultiplier


def test_flatten():
    t_flatten = Flatten[int](
        uid="test-flatten",
        i_complex=In[TList[TList[int]]](
            data=TList[TList[int]](
                [
                    TList[int]([1, 2]),
                    TList[int]([3, 4]),
                ]
            )
        ),
    )
    t_flatten.run()
    assert t_flatten.o_flat == Out[TList[int]](data=TList[int]([1, 2, 3, 4]))


def test_filter_duplicates():
    t_filter_duplicates = FilterDuplicates[int](
        uid="test-filter-duplicates",
        i_list=In[TList[int]](data=TList[int]([1, 1, 1, 2])),
    )
    t_filter_duplicates.run()
    assert t_filter_duplicates.o_list == Out[TList[int]](data=TList[int]([1, 2]))


def test_list_multiplier():
    t_list_multiplier = ListMultiplier[int, str](
        uid="test-list-multiplier",
        i_list=In[TList[int]](data=TList[int]([1, 1, 1, 1, 1])),
        i_scalar=In[str](data="a"),
    )
    t_list_multiplier.run()
    assert t_list_multiplier.o_list == Out[TList[str]](data=TList[str](["a", "a", "a", "a", "a"]))
