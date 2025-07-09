class Type:
    """Tree Type."""

    FIFO = 1
    LIFO = 2


class TreeItem:
    """TreeItem."""

    def __init__(self, data, data_type, parent, parent_index, result, path, extra):
        """__init__."""
        self.data = data
        self.data_type = data_type
        self.parent = parent
        self.parent_index = parent_index
        self.result = result
        self.path = path
        self.extra = extra if extra is not None else {}


class UnknownItemError(Exception):
    """UknownItemError."""

    pass


class Tree:
    """Tree."""

    TYPE = Type.FIFO
    handlers = []

    def __init__(self, result):
        """__init__."""
        self.stack = []
        self.root_result = result
        self.result = None
        self.current = None
        self.parent = None

    def __iter__(self):
        """Iterate over stack."""
        return self

    def __next__(self):
        """Get next item from stack."""
        try:
            stack_item = self.stack.pop(0)
        except IndexError:
            raise StopIteration
        self.current = stack_item
        self.parent = stack_item.parent
        self.process_item(stack_item)
        return stack_item

    def process(self):
        """Process provided stack."""
        for _ in self:
            pass
        return self.result

    def add_to_process(self, data=None, data_type=None, parent_index=None, result=None, extra=None):
        """Add new item to stack to process."""
        if self.TYPE == Type.FIFO:
            titem = TreeItem(
                data=data,
                data_type=data_type,
                parent=self.current,
                parent_index=parent_index,
                result=result,
                path=(
                    self.current.path + "." + str(parent_index)
                    if self.current
                    else str(parent_index)
                ),
                extra=extra,
            )
            self.stack.insert(0, titem)
        else:
            titem = TreeItem(
                data=data,
                data_type=data_type,
                parent=self.current,
                parent_index=parent_index,
                result=result,
                path=(
                    self.current.path + "." + str(parent_index)
                    if self.current
                    else str(parent_index)
                ),
            )
            self.stack.append(titem)

    # def add_to_result(self):
    #     self.result.append(self.current)

    def process_item(self, stack_item):
        """Process stack item."""
        for handler in self.handlers:
            if handler.match(stack_item):
                handler.process(self, stack_item)
                break
        else:
            raise UnknownItemError(stack_item.path, stack_item.data, self.handlers)


class ItemHandler:
    """ItemHandler."""

    def match(self, item: TreeItem):
        """Match."""
        return True

    def process(self, tree: Tree, item: TreeItem):
        """Process."""
        pass
