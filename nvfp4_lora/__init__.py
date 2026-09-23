"""NVFP4 dequantization and LoRA training primitives for DGX Spark sm_121."""
from importlib import import_module

__all__ = [
    "dequantize_nvfp4_weight",
    "NVFP4_E2M1_LUT",
    "quantize_nvfp4_2d",
    "quantize_nvfp4_3d_per_slice",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "dequant" if name in {"dequantize_nvfp4_weight", "NVFP4_E2M1_LUT"} else "quantize"
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
