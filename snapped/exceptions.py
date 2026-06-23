class NotSupportedType(Exception):
    """Exception when the Snapped Code encounters an unexpected runtime type"""

    label: str
    obj: type | str

    def __init__(self, label: str, obj: type | str) -> None:
        self.label = label
        self.obj = obj
        if isinstance(obj, str):
            super().__init__(
                f"{obj.__module__}.{obj} is not supported by the {label} compiler"
            )
        else:
            super().__init__(
                f"{obj.__module__}.{obj.__name__} is not supported by the {label} compiler"
            )


class FailedConsistencyCheck(Exception):
    """
    Exception when the code entered a path it should never have.
    Report these exceptions immediately.
    """

    label: str
    error: str | Exception

    def __init__(self, label: str, error: str | Exception) -> None:
        self.label = label
        self.error = error
        super().__init__(f"Consistency Check Failed in {self.label}: {self.error}")
