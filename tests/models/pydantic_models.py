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

from ._fields import (
    DeviceKind,
    ShortStr,
    CapabilityFlag,
    DataType,
    TensorShape,
    Precision,
    PositiveDec,
    UnitFloat,
)


class MemoryBudget(BaseModel):
    """Immutable memory envelope; available_bytes is computed."""

    model_config = ConfigDict(frozen=True)

    total_bytes: PositiveInt
    reserved_bytes: NonNegativeInt = 0
    swap_bytes: NonNegativeInt = 0
    unified: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_bytes(self) -> int:
        return self.total_bytes - self.reserved_bytes

    @model_validator(mode="after")
    def _reserved_le_total(self) -> MemoryBudget:
        if self.reserved_bytes > self.total_bytes:
            raise ValueError("reserved_bytes cannot exceed total_bytes")
        return self


class DeviceSpec(BaseModel):
    """Full hardware descriptor including NUMA topology and peer-access sets."""

    model_config = ConfigDict(populate_by_name=True)

    kind: DeviceKind
    index: NonNegativeInt = 0
    name: ShortStr | None = None
    # (major, minor) CUDA compute capability, e.g. (9, 0) for H100
    compute_capability: tuple[int, int] | None = None
    capabilities: CapabilityFlag = CapabilityFlag.NONE
    memory: MemoryBudget
    clock_mhz: PositiveFloat | None = None
    numa_node: Annotated[int, Field(ge=0)] | None = None
    peer_devices: frozenset[int] = frozenset()
    env_vars: dict[str, str] = Field(default_factory=dict)

    @field_validator("compute_capability")
    @classmethod
    def _valid_cc(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and v[0] < 1:
            raise ValueError(f"Invalid compute capability major: {v[0]}")
        return v


class TensorSpec(BaseModel):
    """Shape + dtype contract for an edge in the graph. -1 means dynamic dim."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dtype: DataType
    shape: TensorShape
    min_shape: TensorShape | None = None
    max_shape: TensorShape | None = None
    layout: Literal["contiguous", "strided", "sparse_coo", "sparse_csr"] = "contiguous"
    requires_grad: bool = False
    fill_value: float | int | None = None
    tags: frozenset[str] = frozenset()
    description: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rank(self) -> int:
        return len(self.shape)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def numel(self) -> int | None:
        if -1 in self.shape:
            return None
        n = 1
        for d in self.shape:
            n *= d
        return n

    @model_validator(mode="after")
    def _bounds_rank_match(self) -> TensorSpec:
        for attr in ("min_shape", "max_shape"):
            v = getattr(self, attr)
            if v is not None and len(v) != len(self.shape):
                raise ValueError(f"{attr} rank must match shape rank")
        return self


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    precision: Precision = Precision.INT8
    per_channel: bool = True
    symmetric: bool = True
    calibration_samples: PositiveInt = 512
    observer: Literal["minmax", "percentile", "mse", "kl"] = "percentile"
    percentile: UnitFloat = 0.9999
    fake_quant: bool = False
    # stored as compiled re.Pattern objects
    skip_layers: list[re.Pattern[str]] = Field(default_factory=list)

    @field_validator("skip_layers", mode="before")
    @classmethod
    def _compile_patterns(cls, v: list[Any]) -> list[re.Pattern[str]]:
        return [re.compile(p) if isinstance(p, str) else p for p in v]


class NormConfig(BaseModel):
    kind: Literal["batch", "layer", "group", "instance", "rms"] = "layer"
    eps: Annotated[float, Field(gt=0.0, lt=1.0)] = 1e-5
    affine: bool = True
    num_groups: PositiveInt | None = None

    @model_validator(mode="after")
    def _group_needs_groups(self) -> NormConfig:
        if self.kind == "group" and self.num_groups is None:
            raise ValueError("num_groups required for group normalisation")
        return self


class AttentionConfig(BaseModel):
    num_heads: PositiveInt
    head_dim: PositiveInt
    dropout: UnitFloat = 0.0
    causal: bool = False
    window_size: Annotated[int, Field(gt=0)] | None = None
    rope_theta: PositiveFloat = 10_000.0
    rope_scaling: dict[str, float | str] | None = None
    backend: Literal["flash", "memory_efficient", "math"] = "flash"
    kv_cache: bool = False
    softcap: PositiveFloat | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim


class _BaseOptimiser(BaseModel):
    kind: str
    lr: PositiveDec = Decimal("1e-3")
    weight_decay: NonNegativeFloat = 0.0
    grad_clip: PositiveFloat | None = None


class AdamWConfig(_BaseOptimiser):
    kind: Literal["adamw"] = "adamw"  # type: ignore
    betas: tuple[UnitFloat, UnitFloat] = (0.9, 0.999)
    eps: Annotated[float, Field(gt=0.0)] = 1e-8
    amsgrad: bool = False
    fused: bool = True


class SGDConfig(_BaseOptimiser):
    kind: Literal["sgd"] = "sgd"  # type: ignore
    momentum: UnitFloat = 0.9
    dampening: NonNegativeFloat = 0.0
    nesterov: bool = True


class AdaFactorConfig(_BaseOptimiser):
    kind: Literal["adafactor"] = "adafactor"  # type: ignore
    scale_parameter: bool = True
    relative_step: bool = True
    warmup_init: bool = False
    min_dim_size_to_factor: PositiveInt = 128


class LionConfig(_BaseOptimiser):
    kind: Literal["lion"] = "lion"  # type: ignore
    betas: tuple[UnitFloat, UnitFloat] = (0.9, 0.99)
    use_triton: bool = False


OptimizerConfig = Annotated[
    Union[AdamWConfig, SGDConfig, AdaFactorConfig, LionConfig],
    Discriminator("kind"),
]
