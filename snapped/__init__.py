import json
import logging
from ._model import SnappedModel, SupportedLibrary
from ._version import __label__, __version__
from .exceptions import NotSupportedType, FailedConsistencyCheck


def snap(model: type) -> SnappedModel:

    def _dataclass() -> tuple[SupportedLibrary, dict] | None:
        from ._dataclasses import DataclassSnapHandler

        if DataclassSnapHandler.is_proper_type(model):
            return ("dataclass", DataclassSnapHandler.serialize(model))

        return None

    def _pydantic() -> tuple[SupportedLibrary, dict] | None:
        from ._pydantic import PydanticSnapHandler, BaseModel

        if PydanticSnapHandler.is_proper_type(model):
            return ("pydantic", PydanticSnapHandler.serialize(model))

        return None

    def _marshmallow() -> tuple[SupportedLibrary, dict] | None:
        from . import _marshmallow

        return None

    logger = logging.getLogger(__label__)

    for cb in [_dataclass, _pydantic, _marshmallow]:
        try:
            match cb():
                case None:
                    continue
                case (l, s):
                    return SnappedModel(library=l, schema=s)
                case x:
                    raise FailedConsistencyCheck(
                        label="snap",
                        error="Invalid callback type. "
                        f"Callback={cb}; Type={type(x)}",
                    )
        except ImportError as e:
            print(e)
            continue

    logger.error(
        f"There was no supported compiler for the type of {model} "
        " - did you make sure the optional dependencies were installed?"
    )
    raise NotSupportedType(label="snap", obj=model)


def unsnap(model: str | dict | SnappedModel, base: type | None = None) -> type:
    logger = logging.getLogger(__label__)
    if isinstance(model, str):
        model = json.loads(model)
    if isinstance(model, dict):
        model = SnappedModel(**model)
    if isinstance(model, SnappedModel) is False:
        raise RuntimeError(
            f"Invalid object reference passed to snapped.unsnap. Got {type(model)}. "
            "Expected: str | dict | SnappedModel"
        )
    s_model: SnappedModel = model  # type: ignore
    try:
        match s_model.library:
            case "dataclass":
                from ._dataclasses import DataclassSnapHandler

                return DataclassSnapHandler.deserialize(
                    s_model.schema, s_model.version, base=base
                )
            case "pydantic":
                from ._pydantic import PydanticSnapHandler

                return PydanticSnapHandler.deserialize(
                    s_model.schema, s_model.version, base=base
                )
            case "marshmallow":
                raise KeyError("Not supported: 'marshmallow'")
    except ImportError:
        logger.error(
            f"There was no supported compiler for the type of {model} "
            " - did you make sure the optional dependencies were installed?"
        )
        raise NotSupportedType(label="snap", obj=s_model.library)


__all__ = ["snap", "unsnap", "SnappedModel", "__label__", "__version__"]
