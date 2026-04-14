"""EDA helpers for the MRI classification project."""

__all__ = ["EDAAnaLiz"]


def __getattr__(name: str):
    if name == "EDAAnaLiz":
        from .eda_araclar import EDAAnaLiz

        return EDAAnaLiz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
