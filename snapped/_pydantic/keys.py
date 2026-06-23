class PydanticKeys:
    DEFS: str = "$defs"
    REFERENCE: str = "$ref"
    PROPERTIES: str = "properties"
    REQUIRED: str = "required"
    ANY_OF: str = "anyOf"
    ALL_OF: str = "allOf"
    CONST: str = "const"
    ENUM: str = "enum"
    TYPE: str = "type"
    ITEMS: str = "items"
    PREFIX_ITEMS: str = "prefixItems"
    UNIQUE_ITEMS: str = "uniqueItems"
    ARRAY: str = "array"
    OBJECT: str = "object"
    ADD_PROPS: str = "additionalProperties"
    FORMAT: str = "format"
    DATE_TIME: str = "date-time"
    DATE: str = "date"
    UUID: str = "uuid"
    ALIAS: str = "alias"


CONSTRAINT_MAP = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "multipleOf": "multiple_of",
    "minLength": "min_length",
    "maxLength": "max_length",
    "pattern": "pattern",
    "minItems": "min_length",
    "maxItems": "max_length",
}

PRIMITIVE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
}
