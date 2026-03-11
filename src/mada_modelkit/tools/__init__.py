"""Tools package for mada-modelkit.

Tool registry and function calling support.
Zero external dependencies — stdlib only.
"""

from mada_modelkit.tools.registry import Tool, ToolRegistry

__all__ = [
    "Tool",
    "ToolRegistry",
]
