"""Namespace bridge for relocated TempFlow code plus the external AWM checkout."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
