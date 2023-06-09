from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class RFConfig:
    pass
    # TODO


@dataclass
class LogregConfig:
    _target_: str = field(default='sklearn.linear_model.LogisticRegression')
    penalty: str = field(default='l1')
    solver: str = field(default='liblinear')
    C: float = field(default=1.0)
    random_state: int = field(default=42)
    max_iter: int = field(default=100)


@dataclass
class ModelConfig:
    model_name: str
    model_params: Any
