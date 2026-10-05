"""Base của kind `feature`: class `Feature` định nghĩa ở core/component.py (CONTRACTS §3) vì core/execution.py cần nó;
file này chỉ re-export để mọi kind đều có base.py."""

from ...core.component import Feature

__all__ = ["Feature"]
