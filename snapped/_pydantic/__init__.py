import logging
from datetime import datetime, date
from uuid import UUID
from typing import (
    Any,
    Optional,
    Union,
    Literal,
    Tuple,
    Set,
    List,
    Dict,
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field, create_model, fields

# Local Imports
from .keys import PydanticKeys, CONSTRAINT_MAP, PRIMITIVE_MAP
from .._version import __label__
from .._model import ROOT_NAME, BASE_NAME


class PydanticSnapHandler:
    name: str
    """Name of the Base Pydantic Model"""

    registry: dict[str, type[BaseModel]]
    """Registry so $ref lookups can find already-built classes."""

    logger: logging.Logger

    def __init__(self, name: str) -> None:
        self.name = name
        self.registry = {}
        self.logger = logging.getLogger(__label__)

    def _is_set_type(self, py_type: Any) -> bool:
        """Checks if the Python type is a set"""
        return get_origin(py_type) is set

    def is_optional_type(self, py_type: Any) -> bool:
        """Checks if the Python type is nullable"""
        if get_origin(py_type) is Union:
            return type(None) in get_args(py_type)
        return False

    def _infer_default_factory(self, py_type: Any):
        """Maps a Python type to the standard factory"""
        origin = get_origin(py_type)
        if origin is list:
            return list
        if origin is set:
            return set
        if origin is dict:
            return dict
        return None

    def ref_name(self, ref: str) -> str:
        """Rebrand '#/$defs/MyModel' into 'MyModel'"""
        return ref.lstrip("#/$defs/")

    def _topological_sort(self, defs: dict[str, dict]) -> list[str]:
        """
        Return def names in an order where dependencies come before dependents.
        Uses a simple DFS.  Cycles are not expected in valid Pydantic schemas.
        """

        def _find_refs(schema: dict) -> list[str]:
            """Recursively collect all $ref names inside a schema fragment."""
            refs: list[str] = []
            if "$ref" in schema:
                refs.append(self.ref_name(schema["$ref"]))
            for value in schema.values():
                if isinstance(value, dict):
                    refs.extend(_find_refs(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            refs.extend(_find_refs(item))
            return refs

        def _visit(name: str) -> None:
            nonlocal order, visited

            if name in visited:
                return
            visited.add(name)
            defn = defs.get(name, {})
            for dep in _find_refs(defn):
                if dep in defs:
                    _visit(dep)
            order.append(name)

        order: list[str] = []
        visited: set[str] = set()

        for n in defs:
            _visit(n)

        return order

    def _build_model(
        self, name: str, schema: dict, base: Optional[type]
    ) -> type[BaseModel]:
        """Turn a single JSON Schema object node into a BaseModel subclass."""

        properties: dict[str, Any] = schema.get(PydanticKeys.PROPERTIES, {})
        required_set: set[str] = set(schema.get(PydanticKeys.REQUIRED, []))
        field_definitions: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            py_name: str
            alias: str | None = None
            py_type = self._schema_to_type(field_schema)

            # Build Field name
            if x := field_schema.get(PydanticKeys.ALIAS):
                py_name = field_name
                alias = x
            elif not field_name.isidentifier() or (
                field_name != field_name.lower() and "_" not in field_name
            ):
                py_name = field_name.lower()
                alias = field_name
            else:
                py_name = field_name
                alias = None

            field_definitions[py_name] = (
                py_type,
                self._schema_to_field(
                    schema=field_schema,
                    is_required=(field_name in required_set),
                    py_type=py_type,
                    alias=alias,
                ),
            )
        model = create_model(name, __base__=base, **field_definitions)
        if desc := schema.get("description"):
            model.__doc__ = desc
        return model

    def _schema_to_type(self, schema: dict) -> Any:
        """
        Convert a Pydantic field definition to the re-compiled type
        """

        def _union_of(types: list[Any]) -> Any:
            if not types:
                return Any
            if len(types) == 1:
                return types[0]
            # Build Union dynamically
            return Union[tuple(types)]

        def _any_of_to_type(options: list[dict]):
            nulls = [s for s in options if s.get(PydanticKeys.TYPE) == "null"]
            non_nulls = [s for s in options if s.get(PydanticKeys.TYPE) != "null"]

            inner_types = [self._schema_to_type(s) for s in non_nulls]

            match len(inner_types):
                case 0:
                    return type(None)
                case 1:
                    base = inner_types[0]
                case _:
                    base = _union_of(inner_types)

            if nulls:
                return Optional[base]
            return base

        def _object_of(schema: dict):
            props = schema.get(PydanticKeys.ADD_PROPS)
            if props is not None and isinstance(props, dict):
                return Dict[str, self._schema_to_type(props)]
            return Dict[str, Any]

        def _array_of(schema: dict):
            items = schema.get(PydanticKeys.ITEMS)
            prefix_items = schema.get(PydanticKeys.PREFIX_ITEMS)

            if prefix_items is not None:
                return Tuple[tuple(self._schema_to_type(s) for s in prefix_items)]
            item_type = self._schema_to_type(items) if items else Any
            if schema.get(PydanticKeys.UNIQUE_ITEMS):
                return Set[item_type]
            return List[item_type]

        # Case: Pydantic Model Type
        if r := schema.get(PydanticKeys.REFERENCE):
            name: str = self.ref_name(r)
            if m := self.registry.get(name):
                return m
            raise RuntimeError(
                f"{PydanticKeys.REFERENCE}: {name} not yet in registry — check build order"
            )
        # Case: Union
        if u := schema.get(PydanticKeys.ANY_OF):
            return _any_of_to_type(u)

        # Case: Everything
        if a := schema.get(PydanticKeys.ALL_OF):
            if len(a) == 1:
                return self._schema_to_type(a[1])
            return _union_of([self._schema_to_type(s) for s in a])

        # Case: Literals
        if c := schema.get(PydanticKeys.CONST):
            return Literal[c]
        if e := schema.get(PydanticKeys.ENUM):
            return Literal[tuple(e)]

        # Case Dict / Arrays
        match schema.get(PydanticKeys.TYPE):
            case None:
                return Any
            case PydanticKeys.ARRAY:
                return _array_of(schema)
            case PydanticKeys.OBJECT:
                return _object_of(schema)
            case x:
                json_type: str = x

        # Primitives
        if p := PRIMITIVE_MAP.get(json_type):
            return p

        # String Formats
        match schema.get(PydanticKeys.FORMAT):
            case PydanticKeys.DATE_TIME:
                return datetime
            case PydanticKeys.DATE:
                return date
            case PydanticKeys.UUID:
                return UUID
        # Fallback
        return Any

    def _schema_to_field(
        self, schema: dict, is_required: bool, py_type: Any, alias: str | None = None
    ) -> fields.FieldInfo:
        """Builds a Pydantic Field definition from the compiled dict"""
        kwargs: dict[str, Any] = {}

        if a := alias or schema.get("alias"):
            kwargs["alias"] = a

        if d := schema.get("description"):
            kwargs["description"] = d

        if t := schema.get("title"):
            kwargs["title"] = t

        for json_k, py_key in CONSTRAINT_MAP.items():
            if (v := schema.get(json_k)) is not None:
                kwargs[py_key] = v

        if schema.get("readOnly"):
            kwargs["json_schema_extra"] = {"readOnly": True}

        if not is_required:
            if "default" in schema:
                d = schema["default"]
                if d == [] and self._is_set_type(py_type):
                    kwargs["default_factory"] = set
                elif d == [] or (isinstance(d, list) and not d):
                    kwargs["default_factory"] = list
                elif d == {} or (isinstance(d, dict) and not d):
                    kwargs["default_factory"] = dict
                else:
                    kwargs["default"] = d
            else:
                if f := self._infer_default_factory(py_type):
                    kwargs["default_factory"] = f
                elif self.is_optional_type(py_type):
                    kwargs["default"] = None

        return Field(**kwargs)

    def build(self, schema: dict, base: Optional[type]) -> type[BaseModel]:
        """
        Compiles the root Pydantic Model
        """
        defs: dict[str, dict] = schema.get(PydanticKeys.DEFS, {})
        for cls in self._topological_sort(defs):
            self.logger.debug(f"Builmsg=ding class definition {cls}")
            self.registry[cls] = self._build_model(cls, defs[cls], base=None)
        self.logger.info(f"Building class definition {self.name}")
        return self._build_model(self.name, schema, base=base)

    @staticmethod
    def is_proper_type(model: Any) -> bool:
        return issubclass(model, BaseModel)

    @staticmethod
    def serialize(model: type[BaseModel]) -> dict:
        return model.model_json_schema(mode="serialization") | {
            ROOT_NAME: model.__name__,
            BASE_NAME: model.__base__.__name__ if model.__base__ else None,
        }

    @staticmethod
    def deserialize(
        schema: dict, version: tuple[int, int], base: Optional[type]
    ) -> type[BaseModel]:
        return PydanticSnapHandler(name=schema.get(ROOT_NAME, "Model")).build(
            schema, base=base
        )
