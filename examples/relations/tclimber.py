from typing import List, Dict, Union, Any, Callable
from dataclasses import dataclass

@dataclass
class Leaf:
    pass


class TResult:
    def __init__(self, result):
        pass

    def on_branch(self, parent, parent_key, current, stack):
        return self

    def on_leaf(self, k, parent, v, stack) -> Any:
        return None

    def add_to_stack(self, stack, parent, key, to_stack):
        stack.append((parent, key, to_stack))


def default_list_matcher(json):
    return isinstance(json, list)

def default_dict_matcher(json):
    return isinstance(json, dict)


@dataclass
class Tree:
    children: Dict[Union[str, int], 'Tree']
    list_matcher: Callable[[Any], bool] = default_list_matcher
    dict_matcher: Callable[[Any], bool] = default_dict_matcher

    @classmethod
    def from_json(cls, json: Union[Dict[str, Any], List[Any]]):
        ret = cls(children={})
        stack = [(ret, 'root', json)]

        while stack:
            parent, key, current_json = stack.pop()
            if cls.list_matcher(json):
                for n, x in enumerate(json):
                    parent.children[n] = Tree(children={}, list_matcher=self.list_matcher, dict_matcher=self.dict_matcher)
                    stack.append((parent.children[n], n, x))
            if cls.dict_matcher(json):
                for k, v in json.items():
                    parent.children[k] = (Tree(children={}, list_matcher=self.list_matcher, dict_matcher=self.dict_matcher))
                    stack.append((parent.children[k], k, v))
            else:
                parent.children[key] = json
        ret.children = ret.children['root']
        return ret

    def harvest(self, result: TResult):
        # stack = [(parent, parent_key, current)]
        stack = [(None, None, self)]
        while stack:
            parent, key, current = stack.pop()
            if isinstance(current, self.__class__):
                nresult = result.on_branch(parent, key, current, stack)
                for k, v in current.children.items():
                    stack.append((current, k, v))
            else:
                result.on_leaf(key, parent, current, stack)
        print("----")

    def to_json(self, result: TResult):
        _result = {}
        stack = [(_result, 'root', self)]
        while stack:
            parent, key, current = stack.pop()
            print("KEY", key, current)
            parent[key] = {}
            if isinstance(current, self.__class__):
                for k, v in current.children.items():
                    print("KV", k, v)
                    stack.append((parent[key], k, v))
            else:
                ret = result.on_leaf(key, parent, current, stack)
                print(ret)
                parent[key] = ret
        return _result['root']
