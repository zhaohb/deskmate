"""Local Ollama service provisioning + lifecycle (Model Service page).

See :mod:`deskmate.modelsvc.service` for the implementation. This package wraps
downloading/launching the Ollama service so the API and daemon can drive it
without knowing the platform details.
"""

from __future__ import annotations

from .service import (
    BACKEND_OFFICIAL,
    BACKEND_OPENVINO,
    GENAI_RUNTIME_URL,
    OFFICIAL_URL,
    build_launch_env,
    download_genai,
    download_zip,
    extract_zip,
    find_exe_in_dir,
    install_genai_runtime,
    install_official,
    list_genai_versions,
    obtain_openvino_exe,
    pull_model_stream,
    read_service_log,
    resolve_download_dir,
    resolve_exe,
    start_service,
    status,
    stop_service,
    validate_exe_path,
)

__all__ = [
    "BACKEND_OFFICIAL",
    "BACKEND_OPENVINO",
    "GENAI_RUNTIME_URL",
    "OFFICIAL_URL",
    "build_launch_env",
    "download_genai",
    "download_zip",
    "extract_zip",
    "find_exe_in_dir",
    "install_genai_runtime",
    "install_official",
    "list_genai_versions",
    "obtain_openvino_exe",
    "pull_model_stream",
    "read_service_log",
    "resolve_download_dir",
    "resolve_exe",
    "start_service",
    "status",
    "stop_service",
    "validate_exe_path",
]
