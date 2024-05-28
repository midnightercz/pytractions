from pytractions.base import (
    TList,
    Out,
    In,
)

from pytractions.transformations import Flatten, FilterDuplicates, ListMultiplier


def test_flatten():
    t_flatten = Flatten[int](
        uid="test-flatten",
        i_complex=In[TList[In[TList[In[int]]]]](
            data=TList[In[TList[In[int]]]](
                [
                    In[TList[In[int]]](data=TList[In[int]]([In[int](data=1), In[int](data=2)])),
                    In[TList[In[int]]](data=TList[In[int]]([In[int](data=3), In[int](data=4)])),
                ]
            )
        ),
    )
    t_flatten.run()
    print(t_flatten.o_flat)
    assert t_flatten.o_flat == Out[TList[Out[int]]](
        data=TList[Out[int]](
            [Out[int](data=1), Out[int](data=2), Out[int](data=3), Out[int](data=4)]
        )
    )


def test_filter_duplicates():
    t_filter_duplicates = FilterDuplicates[int](
        uid="test-filter-duplicates",
        i_list=In[TList[In[int]]](
            data=TList[In[int]]([In[int](data=1), In[int](data=2), In[int](data=1)])
        ),
    )
    t_filter_duplicates.run()
    print(t_filter_duplicates.o_list)
    assert t_filter_duplicates.o_list == Out[TList[Out[int]]](
        data=TList[Out[int]]([Out[int](data=1), Out[int](data=2)])
    )


def test_list_multiplier():
    t_list_multiplier = ListMultiplier[int, str](
        uid="test-list-multiplier",
        i_list=In[TList[int]](data=TList[int]([1, 2, 3, 4, 5])),
        i_scalar=In[str](data="a"),
    )
    t_list_multiplier.run()
    assert t_list_multiplier.o_list == Out[TList[str]](data=TList[str](["a", "a", "a", "a", "a"]))
