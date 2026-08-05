"""
Core package exposing modularized components of the Gran Sabio LLM Engine.
Importing this package ensures all route modules are loaded so route
definitions attach to the shared FastAPI application.
"""

# Import order matters: ensure app state is initialized before routes.
# Route modules register themselves upon import.
# Path-shape conflicts are NOT resolved here: streaming_routes imports
# console_routes explicitly so that the literal /stream/console is always
# registered before the /stream/{session_id} catch-all, whatever this list says.
from . import (
    analytics_routes,  # noqa: F401
    app_state,  # noqa: F401
    console_routes,  # noqa: F401
    context_routes,  # noqa: F401
    debug_routes,  # noqa: F401
    generation_routes,  # noqa: F401
    interfaces,  # noqa: F401
    openrouter_routes,  # noqa: F401
    streaming_routes,  # noqa: F401
)
