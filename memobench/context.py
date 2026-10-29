"""Per-thread request metadata (system, conv_id, question_id, stage) attached to every logged call.

Worker threads call set_meta(...) at the start of each task; the Memora adapter reads get_meta().
"""
import threading

_local = threading.local()


def set_meta(**kw) -> None:
    _local.meta = {**getattr(_local, "meta", {}), **kw}


def clear_meta() -> None:
    _local.meta = {}


def get_meta() -> dict:
    return dict(getattr(_local, "meta", {}))
