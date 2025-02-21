"""Top-level package for inferencescale."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Max Clouser"""
__email__ = "maximclouser@gmail.com"
__version__ = "0.0.1"

from .src.inferencescale.nodes import NODE_CLASS_MAPPINGS
from .src.inferencescale.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
