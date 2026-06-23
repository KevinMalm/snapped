__label__ = "snapped"
try:
    from importlib.metadata import version

    __version__ = version(__label__)
except:
    __version__ = "pre-release"
