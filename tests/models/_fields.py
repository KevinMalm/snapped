import re
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, IntEnum, IntFlag, auto
from ipaddress import IPv4Address
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    NewType,
    TypeVar,
    Union,
)
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)

# ── NewType aliases ────────────────────────────────────────────────────────────
NodeId = NewType("NodeId", str)
EdgeId = NewType("EdgeId", str)
TensorShape = tuple[int, ...]

# ── TypeVars ───────────────────────────────────────────────────────────────────
T = TypeVar("T")

# ── Constrained scalar aliases ─────────────────────────────────────────────────
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveDec = Annotated[Decimal, Field(gt=Decimal("0"))]
ShortStr = Annotated[str, Field(min_length=1, max_length=64)]
SemVer = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]
SHA256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OrcidStr = Annotated[str, Field(pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")]
PortNumber = Annotated[int, Field(ge=1, le=65535)]
PriorityInt = Annotated[int, Field(ge=0, le=100)]


class DataType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    STRING = "string"
    BYTES = "bytes"


class DeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    TPU = "tpu"
    MPS = "mps"  # Apple Silicon


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class ActivationFn(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


class NodeRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    BRANCH = "branch"
    MERGE = "merge"
    CHECKPOINT = "checkpoint"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class SchedulePolicy(IntEnum):
    FIFO = 1
    LIFO = 2
    PRIORITY = 3
    ROUND_ROBIN = 4
    WORK_STEALING = 5


class CapabilityFlag(IntFlag):
    NONE = 0
    FP16 = auto()
    INT8 = auto()
    TENSOR_CORES = auto()
    SPARSITY = auto()
    NVLINK = auto()
    PEER_ACCESS = auto()
    ALL = FP16 | INT8 | TENSOR_CORES | SPARSITY | NVLINK | PEER_ACCESS
