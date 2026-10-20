"""CPU-only lifecycle checks; no subprocess, HTTP request, or model execution."""

import importlib.util
import signal
import subprocess
import unittest
from pathlib import Path

P = Path(__file__).with_name("run_when_qualified.py")
spec = importlib.util.spec_from_file_location("supervisor", P)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Child:
    def __init__(self, stops_on=None, signal_during_wait=False):
        self.stops_on = stops_on
        self.signals = []
        self.returncode = None
        self.signal_during_wait = signal_during_wait

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def wait(self, timeout):
        if self.signal_during_wait:
            s.interrupted(signal.SIGTERM, None)
        if self.signals[-1] == self.stops_on:
            self.returncode = -int(self.stops_on)
            return self.returncode
        raise subprocess.TimeoutExpired("owned-child-fixture", timeout)


class Tests(unittest.TestCase):
    def setUp(self):
        s.child = None
        s.child_record = {"process_id": "fixture"}
        s.interrupt_requested = None
        s.state = {"original_models": ["original"], "cleanup_errors": []}
        s.save = lambda: None
        s.original = object()
        self.posts = []
        s.request = lambda *a, **k: self.posts.append(a)
        s.idle = lambda deadline: True
        s.wait_headroom = lambda *a: None
        s.models = lambda deadline: [
            {"id": "original", "loaded": False, "is_loading": False}
        ]

    def test_handler_remembers_without_raising(self):
        s.interrupted(signal.SIGINT, None)
        s.interrupted(signal.SIGTERM, None)
        self.assertEqual(s.interrupt_requested, signal.SIGINT)
        with self.assertRaises(KeyboardInterrupt):
            s.check_interrupt()

    def test_child_interrupt_clean_exit(self):
        child = s.child = Child(signal.SIGINT)
        s.stop_child()
        self.assertIsNone(s.child)
        self.assertEqual(child.signals, [signal.SIGINT])
        self.assertEqual(s.child_record["exit_code"], -signal.SIGINT)

    def test_exact_bounded_escalation(self):
        child = s.child = Child(signal.SIGKILL)
        s.stop_child()
        self.assertIsNone(s.child)
        self.assertEqual(child.signals, [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(
            [r["grace_seconds"] for r in s.child_record["shutdown_signals"]],
            [90, 30, 30],
        )

    def test_second_signal_does_not_escape_cleanup(self):
        s.child = Child(signal.SIGINT, signal_during_wait=True)
        s.stop_child()
        self.assertIsNone(s.child)
        self.assertEqual(s.interrupt_requested, signal.SIGTERM)

    def test_uncertain_child_prevents_model_restore(self):
        s.child = Child()
        with self.assertRaises(RuntimeError):
            s.stop_child()
        s.restore_original_models()
        self.assertFalse(s.state["restoration_complete"])
        self.assertEqual(self.posts, [])

    def test_low_memory_prevents_load(self):
        def low(*args):
            raise MemoryError("fixture")

        s.wait_headroom = low
        s.restore_original_models()
        self.assertFalse(s.state["restoration_complete"])
        self.assertEqual(self.posts, [])

    def test_busy_original_prevents_load(self):
        s.idle = lambda deadline: False
        s.restore_original_models()
        self.assertFalse(s.state["restoration_complete"])
        self.assertEqual(self.posts, [])

    def test_new_user_model_prevents_load(self):
        s.models = lambda deadline: [
            {"id": "original", "loaded": False, "is_loading": False},
            {"id": "new", "loaded": True, "is_loading": False},
        ]
        s.restore_original_models()
        self.assertFalse(s.state["restoration_complete"])
        self.assertEqual(self.posts, [])

    def test_successful_restoration(self):
        s.models = lambda deadline: [
            {"id": "original", "loaded": bool(self.posts), "is_loading": False}
        ]
        s.restore_original_models()
        self.assertTrue(s.state["restoration_complete"])
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0][1], "/admin/api/models/original/load")

    def test_already_restored_does_not_reload(self):
        s.models = lambda deadline: [
            {"id": "original", "loaded": True, "is_loading": False}
        ]
        s.restore_original_models()
        self.assertTrue(s.state["restoration_complete"])
        self.assertEqual(self.posts, [])


if __name__ == "__main__":
    unittest.main()
