"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str
    table_a: str
    table_b: str
    train: str
    valid: str
    test: str
    id_col: str = "id"
    text_col: str = "title"
    structured_cols: list[str] = field(default_factory=lambda: ["manufacturer", "price"])


@dataclass
class ModelConfig:
    type: str = "lightgbm"
    params: dict = field(default_factory=dict)
    early_stopping_rounds: int = 50


@dataclass
class BlockingConfig:
    method: str = "tfidf_cosine"
    threshold: float = 0.1


@dataclass
class PipelineConfig:
    random_seed: int = 42
    embedding_model: str = "all-MiniLM-L6-v2"
    output_dir: str = "results"
    artifacts_dir: str = "artifacts"


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    blocking: BlockingConfig
    pipeline: PipelineConfig


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw.get("model", {})),
        blocking=BlockingConfig(**raw.get("blocking", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
    )
