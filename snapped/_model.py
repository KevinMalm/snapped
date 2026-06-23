from typing import Literal
from dataclasses import dataclass, field, asdict

ROOT_NAME = "&root"
BASE_NAME = "&base"

SupportedLibrary = Literal["dataclass", "pydantic", "marshmallow"]


@dataclass
class SnappedModel:
    library: SupportedLibrary
    schema: dict
    version: tuple[int, int] = field(default=(0, 0))

    def to_dict(self) -> dict:
        return asdict(self)
