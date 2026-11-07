"""CPU-only test setup.

`nanovllm/__init__.py` imports the model runner (torch, triton, flash-attn), which needs a CUDA GPU.
The scheduler, block manager and sequence code is plain Python, so register a bare `nanovllm`
package that skips that __init__ and lets the tests import those modules directly.
"""
import sys
import types
from pathlib import Path

if "nanovllm" not in sys.modules:
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(Path(__file__).resolve().parent.parent / "nanovllm")]
    sys.modules["nanovllm"] = package

try:
    import transformers  # noqa: F401
except ImportError:
    # nanovllm/config.py imports AutoConfig and evaluates `AutoConfig | None` at class definition,
    # so the stub must be a class. The scheduler only uses Config as a type hint.
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = type("AutoConfig", (), {})
    sys.modules["transformers"] = transformers
