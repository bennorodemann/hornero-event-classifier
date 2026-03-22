from threading import Event


class ComplexEvent:
    # TODO: could be improved but is also good enough
    def __init__(self) -> None:
        self._activate_event = Event()
        self._deactivate_event = Event()
        self._deactivate_event.set()

    def set(self):
        self._activate_event.set()
        self._deactivate_event.clear()

    def is_set(self) -> bool:
        return self._activate_event.is_set()

    def clear(self):
        self._deactivate_event.set()
        self._activate_event.clear()

    def wait_for_set(self, timeout: float | None = None):
        self._activate_event.wait(timeout=timeout)

    def wait_for_clear(self, timeout: float | None = None):
        self._deactivate_event.wait(timeout=timeout)
