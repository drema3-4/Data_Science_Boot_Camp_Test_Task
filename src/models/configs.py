from dataclasses import dataclass
from fastembed.common.model_description import (
    ModelSource,
    PoolingType,
)

@dataclass(frozen=True)
class SparseEmbedderConfig:
    model_name: str
    language: str
    k: float
    b: float
    avg_len: float

@dataclass(frozen=True)
class DenseEmbedderConfig:
    model: str
    sources: ModelSource
    dim: int
    model_file: str
    pooling: PoolingType = PoolingType.MEAN
    normalization: bool = True

@dataclass(frozen=True)
class RerankerConfig:
    name: str
    max_length: int
    device: str