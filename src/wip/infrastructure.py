from collections.abc import Callable

from typing import Generic, TypeVar


T = TypeVar("T")


class DefaultAttr(Generic[T]):
    """Descriptor for default-valued attributes

    Returns a default value (from factory) when the underlying
    private attribute is `None` or missing, and stores values
    set by the user.

    Attributes:
        default_factory: Callable producing a default value.
        init_on_query: If `True`, initializes the attribute with
            the default value on first query if not set. Otherwise,
            just returns the default value without setting the attribute
            (so a new default value will be generated on the next query).
        check: Whether to check if set values match `check_type`.
        check_type: Type to check set values against (if `check` is True).
            If `None`, use the type returned by `default_factory`.
    """

    def __init__(
        self,
        default_factory: Callable[[], T],
        init_on_query: bool = True,
        check: bool = False,
        check_type: type | None = None,
    ) -> None:
        self.default_factory = default_factory
        self.init_on_query = init_on_query
        self.check = check

        if check_type is None:
            check_type = type(default_factory())
        self.check_type = check_type

    def __set_name__(self, owner, name: str) -> None:
        self.private_name = f"_{name}"

    def __get__(self, instance, owner=None) -> T:
        if instance is None:
            return self  # type: ignore[return-value]

        val = getattr(instance, self.private_name, None)
        if val is None:
            val = self.default_factory()
    
            if self.init_on_query:
                setattr(instance, self.private_name, val)

        return val

    def __set__(self, instance, value: T | None) -> None:
        if self.check and not isinstance(value, self.check_type):
            raise TypeError(f"Expected value of type {self.check_type}, got {type(value)}")
        setattr(instance, self.private_name, value)
