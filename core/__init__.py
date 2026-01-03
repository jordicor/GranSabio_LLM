"""
Core package exposing modularized components of the Gran Sabio LLM Engine.
Importing this package ensures all route modules are loaded so route
definitions attach to the shared FastAPI application.
"""

# Import order matters: ensure app state is initialized before routes.
from . import app_state  # noqa: F401

# Route modules register themselves upon import.
from . import interfaces  # noqa: F401
from . import debug_routes  # noqa: F401
from . import generation_routes  # noqa: F401
from . import streaming_routes  # noqa: F401
from . import analytics_routes  # noqa: F401
from . import openrouter_routes  # noqa: F401
