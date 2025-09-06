"""Strategy implementations for building report generation jobs.

The submodule exposes concrete ``JobBuilder`` implementations that
construct reporting workflows from different input scenarios.
"""

from .base import JobBuilder
from .folder import FolderJobBuilder
from .files import FilesJobBuilder
from .multifolder import MultiFolderJobBuilder

__all__ = [
    "JobBuilder",
    "FolderJobBuilder",
    "FilesJobBuilder",
    "MultiFolderJobBuilder",
]
