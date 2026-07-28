"""Registry imports; light mode is reserved for isolated tensor tests."""
import os

if os.environ.get("RSPROMPTER_LIGHT_IMPORT", "0") != "1":
    from .models import *  # noqa: F401,F403
    from .models_sam2 import *  # noqa: F401,F403
