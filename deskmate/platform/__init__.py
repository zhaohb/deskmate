"""Platform-specific capability layer (Intel/Windows power & scheduling).

Additive and zero-invasive: nothing here is imported by existing feature code.
The daemon starts a single PowerManager thread that tags *other* worker threads
with EcoQoS by thread name — the workers themselves are never modified.

Everything degrades gracefully on non-Windows / older platforms: capability
probes return False and all calls become no-ops (never raise).
"""

from .battery import PowerSource, power_source
from .cores import CoreLoad, core_load
from .power_manager import PowerManager
from .processes import AppInfo, AppPowerController, list_apps
from .qos import qos_available, set_thread_eco, set_thread_high

__all__ = [
    "AppInfo",
    "AppPowerController",
    "CoreLoad",
    "PowerManager",
    "PowerSource",
    "core_load",
    "list_apps",
    "power_source",
    "qos_available",
    "set_thread_eco",
    "set_thread_high",
]
