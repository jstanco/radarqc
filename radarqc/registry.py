from typing import Any


class ClassRegistryError(Exception):
    """Custom exception handling case where multiple classes have been
    registered to handle deserializing the same block tag"""

    def __init__(self, parent: type, subclass: type, tag: str) -> None:
        message = (
            "Cannot register more than one {} subclass for tag '{}'.".format(
                parent.__name__, tag
            )
        )
        super().__init__(message)
        self.parent = parent
        self.subclass = subclass
        self.tag = tag


class ClassRegistry:
    def __init__(self) -> None:
        self._registry = {}

    def register(self, tag: str, parent: type, subclass: type) -> None:
        try:
            self._registry[tag]
        except KeyError:
            self._registry[tag] = subclass
        else:
            raise ClassRegistryError(
                parent=parent,
                subclass=subclass,
                tag=tag,
            )

    def create(self, tag, default: type) -> Any:
        cls = self._registry.get(tag, default)
        return cls()
