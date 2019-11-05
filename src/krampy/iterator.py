import itertools
from collections import namedtuple

from .iotools import load_dict_from_file


class ProductIterator(object):
    """An iterator, wrapped around itertools.product, returning namedtuple objects."""

    def __init__(self, *args, from_dict=None, ignore_dict=None, **kwargs):
        self.items = {}
        if from_dict:
            self.items.update(from_dict)
        elif args:
            self.items.update(dict(args))
        elif kwargs:
            self.items.update(dict(**kwargs))

        self.ignore_items = {}
        if ignore_dict:
            self.ignore_items = ProductIterator(from_dict=ignore_dict)

    def __contains__(self, item):
        return any(it == item for it in iter(self))

    def __iter__(self):
        keys = self.items.keys()
        values = [v if hasattr(v, "__iter__") else [v] for v in self.items.values()]
        iterator = itertools.product(*values)

        class_ = namedtuple("IteratorItem", keys)
        for it in iterator:
            if it not in self.ignore_items:
                yield class_(*it)


class Iterator(object):
    def __init__(self, *args, from_dict=None, from_dict_file=None):

        self.sub_iterators = []
        if from_dict:
            self.load_from_dict(from_dict)
        elif from_dict_file:
            dict_ = load_dict_from_file(from_dict_file)
            self.load_from_dict(dict_)
        elif len(args) > 0:
            iterator = ProductIterator(*args)
            self.sub_iterators.append(iterator)

    def load_from_dict(self, dict_):
        # Process dicts in iterDicts list separately
        if "iter_dicts" in dict_:
            for iter_dict in dict_["iter_dicts"]:
                iterator = Iterator(from_dict=iter_dict)
                self.sub_iterators.extend(iterator.sub_iterators)
            dict_.pop("iter_dicts")

        if "product_dict" in dict_:
            iterator = ProductIterator(
                from_dict=dict_["product_dict"], ignore_dict=dict_.get("ignore_dict")
            )
            self.sub_iterators.append(iterator)

    def __iter__(self):
        """Recursively iterate through all iterators."""
        for iterator in self.sub_iterators:
            for item in iterator:
                yield item
