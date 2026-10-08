"""Pure HTTP check preconditions; no native imports or server startup."""


def require_idle_empty(status):
    """Refuse existing observations, processing, capture, meshes and fault state."""
    occupied = {name: status.get(name) for name in (
        "stored_count", "frame_count", "unprocessed_count", "has_mesh",
        "volume_requires_reset", "input_requires_reset", "capture_active",
        "capture_running", "is_capturing", "fusion_failure", "input_failure",
        "fusion_paused") if status.get(name)}
    if status.get("operation") is not None:
        occupied["operation"] = status["operation"]
    if occupied:
        raise RuntimeError(f"Check requires an idle, healthy, empty server before any writes: {occupied}")
    if not status.get("session_id"):
        raise RuntimeError("Server did not provide a session ID for cleanup ownership")


def restore_settings(client, initial, check_session_id):
    """Reset only the acknowledged test session or unchanged initial empty one.

    Invoke against a dedicated check server: the HTTP protocol does not offer an
    atomic compare-and-reset operation for competing clients.
    """
    response = client.get("/api/scan/status")
    response.raise_for_status()
    current = response.json()
    owner = check_session_id if check_session_id is not None else initial["session_id"]
    if current.get("session_id") != owner:
        raise RuntimeError("Server session changed outside this check; refusing cleanup reset")
    if check_session_id is None:
        require_idle_empty(current)
    reset = client.post("/api/scan/reset", json=initial["settings"])
    reset.raise_for_status()
    restored_response = client.get("/api/scan/status")
    restored_response.raise_for_status()
    restored = restored_response.json()
    require_idle_empty(restored)
    if restored["settings"] != initial["settings"]:
        raise RuntimeError("Server settings were not restored after HTTP check")
    return restored
