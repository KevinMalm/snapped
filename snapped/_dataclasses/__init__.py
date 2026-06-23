import logging
from typing import Any, Optional
from dataclasses import make_dataclass, is_dataclass, Field, field
from pydantic import TypeAdapter
from .._model import ROOT_NAME, BASE_NAME
from .._version import __label__
from .._pydantic import PydanticSnapHandler


class DataclassSnapHandler(PydanticSnapHandler):
    name: str
    """Name of the Base Model"""

    registry: dict[str, type]
    """Registry so $ref lookups can find already-built classes."""

    logger: logging.Logger

    def __init__(self, name: str) -> None:
        self.name = name
        self.registry = {}
        self.logger = logging.getLogger(__label__)

    def _schema_to_field_spec(
        self, schema: dict, is_required: bool, py_type: type
    ) -> Field | None:
        if is_required:
            return None
        if r := schema.get("default"):
            if r == [] and self._is_set_type(py_type):
                return field(default_factory=set)  # type: ignore
            if r == [] or (isinstance(r, list) and not r):
                return field(default_factory=list)  # type: ignore
            if r == {} or (isinstance(r, dict) and not r):
                return field(default_factory=dict)  # type: ignore

            return field(default=r)  # type: ignore

        factory = self._infer_default_factory(py_type)
        if factory is not None:
            return field(default_factory=factory)  # type: ignore
        if self.is_optional_type(py_type):
            return field(default=None)

        return field(default=None)

    def _build_model(self, name: str, schema: dict, base: Optional[type]) -> type:
        properties: dict[str, Any] = schema.get("properties", {})
        required_set: set[str] = set(schema.get("required", []))
        required_fields: list[tuple] = []
        optional_fields: list[tuple] = []

        for prop_name, field_schema in properties.items():
            py_type = self._schema_to_type(field_schema)
            is_required = prop_name in required_set
            field_spec = self._schema_to_field_spec(field_schema, is_required, py_type)

            if field_spec is None:
                # Truly required — no default at all
                required_fields.append((prop_name, py_type))
            else:
                optional_fields.append((prop_name, py_type, field_spec))

        return make_dataclass(name, required_fields + optional_fields)

    def build(self, schema: dict, base: Optional[type]) -> type:
        defs: dict[str, dict] = schema.get("$defs", {})

        for name in self._topological_sort(defs):
            self.registry[name] = self._build_model(name, defs[name], base=None)

        return self._build_model(self.name, schema, base=base)

    @staticmethod
    def is_proper_type(model: Any) -> bool:
        return is_dataclass(model)

    @staticmethod
    def serialize(model: type[object]) -> dict:
        if not DataclassSnapHandler.is_proper_type(model):
            raise TypeError(f"{model:!r} is not a dataclass definition")
        return TypeAdapter(model).json_schema() | {
            ROOT_NAME: model.__name__,
            BASE_NAME: (
                f"{model.__base__.__module__}.{model.__base__.__name__}"
                if model.__base__
                else None
            ),
        }

    @staticmethod
    def deserialize(
        schema: dict, version: tuple[int, int], base: Optional[type]
    ) -> type:
        return DataclassSnapHandler(name=schema.get(ROOT_NAME, "Model")).build(
            schema, base=base
        )
