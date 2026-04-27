"""EDA helpers for the MRI classification project."""

__all__ = ["EDAAnaliz", "EDAAnaLiz"]


def __getattr__(name: str):
    if name in {"EDAAnaliz", "EDAAnaLiz"}:
        from .eda_araclar import EDAAnaliz

        return EDAAnaliz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
