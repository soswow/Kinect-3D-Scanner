"""Recovery actions keep the current scan and saved-project identity intact."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

from kinect_scanner.gui import main_window
from kinect_scanner.gui.capture_recovery import CaptureRecoveryDialog
from kinect_scanner.server_task_worker import ServerTaskType
from tests import test_scanner_workflow as workflow
from PyQt6.QtWidgets import QApplication


class CaptureRecoveryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    setUp = workflow.ScannerWorkflowTests.setUp
    tearDown = workflow.ScannerWorkflowTests.tearDown
    fresh_frame = workflow.ScannerWorkflowTests.fresh_frame
    retain_scan = workflow.ScannerWorkflowTests.retain_scan
    task_types = workflow.ScannerWorkflowTests.task_types

    def row(self, **changes):
        return {"path": "/local/retained", "session_id": "retained", "count": 3,
                "retry_safe": True, "error": "", **changes}

    def test_recovery_is_available_offline_and_does_not_upload_on_discovery(self):
        self.window.server_client._connected = False
        self.window._on_recovery_list([self.row()])
        self.assertTrue(self.window.recover_captures_action.isEnabled())
        self.assertIn("(3)", self.window.recover_captures_action.text())
        self.window._recover_captures()
        self.assertEqual([ServerTaskType.LIST_RECOVERY], self.task_types())
        dialog = self.window._recovery_dialog
        self.assertFalse(dialog.retry_button.isEnabled())
        self.assertTrue(dialog.save_button.isEnabled())

    def test_retry_requires_matching_scan_and_updated_server(self):
        self.window.server_client.session_id = "other"
        self.window.server_client._capture_receipts_supported = True
        self.window._on_recovery_list([self.row()])
        self.assertFalse(self.window._retry_captures("/local/retained"))
        self.window.server_client.session_id = "retained"
        self.window.server_client._capture_receipts_supported = False
        self.assertFalse(self.window._retry_captures("/local/retained"))
        self.window.server_client._capture_receipts_supported = True
        self.retain_scan()
        self.assertTrue(self.window._retry_captures("/local/retained"))
        self.assertTrue(self.window._paused)
        self.assertTrue(self.window._recovery_pending)
        self.assertEqual([ServerTaskType.RETRY_CAPTURES], self.task_types())
        self.window._shortcut_pause()
        self.window._shortcut_capture()
        self.window._start_scan(protected=True)
        self.window._reset_scan(protected=True)
        self.window._toggle_connection()
        self.assertEqual([ServerTaskType.RETRY_CAPTURES], self.task_types())
        self.window._on_task_failed("RETRY_CAPTURES", "Connection lost")
        self.assertFalse(self.window._recovery_pending)
        self.assertTrue(self.window._paused)

    def test_offline_partial_save_preserves_current_saved_project_state(self):
        self.retain_scan()
        self.window._project_path = "/local/full.zip"
        self.window._capture_revision = 7
        self.window._saved_revision = 4
        self.window.server_client._connected = False
        self.window._on_recovery_list([self.row(retry_safe=False)])
        with patch.object(main_window.QMessageBox, "information", return_value=main_window.QMessageBox.StandardButton.Save) as explanation, \
                patch.object(main_window.QFileDialog, "getSaveFileName", return_value=("/local/recovery-new.zip", "")):
            self.assertTrue(self.window._save_recovery_project("/local/retained"))
        self.assertIn("partial", explanation.call_args.args[2])
        self.assertEqual([ServerTaskType.EXPORT_RECOVERY], self.task_types())
        self.window._on_recovery_export_done("/local/recovery-new.zip")
        self.assertEqual("/local/full.zip", self.window._project_path)
        self.assertEqual(7, self.window._capture_revision)
        self.assertEqual(4, self.window._saved_revision)
        self.assertTrue(self.window._session_dirty)
        self.assertTrue(self.window._paused)

    def test_retained_barrier_blocks_finish_and_compaction_until_recovery(self):
        self.retain_scan()
        self.window.task_worker.capture_recovery_required = True
        self.window._has_mesh = True
        self.window.server_client._compact_project_supported = True
        self.window._on_recovery_list([self.row()])
        self.assertFalse(self.window.btn_stop_build.isEnabled())
        self.assertFalse(self.window.btn_export_session.isEnabled())
        self.assertFalse(self.window.btn_compact_project.isEnabled())
        self.assertTrue(self.window.recover_captures_action.isEnabled())
        self.window._stop_and_build()
        self.window._pause_or_resume()
        self.assertEqual([], self.task_types())

    def test_corrupt_manifest_is_visible_without_retry_or_silent_export(self):
        dialog = CaptureRecoveryDialog(self.window)
        dialog.set_rows([self.row(session_id=None, error="Unreadable capture buffer manifest")], "retained", True)
        self.assertIn("Unreadable", dialog.detail.text())
        self.assertIn(self.row()["path"], dialog.detail.text())
        self.assertEqual(dialog.detail.accessibleName(), "Capture recovery details")
        self.assertFalse(dialog.retry_button.isEnabled())
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.close()

    def test_opened_partial_recovery_has_persistent_notice(self):
        self.window._restore_server_session({"session_id": "partial", "stored_count": 3,
            "settings": {}, "project_recovery": {"partial": True}})
        self.assertFalse(self.window.recovery_project_notice.isHidden())
        self.assertIn("server-only", self.window.recovery_project_notice.text())
        self.window._restore_server_session({"session_id": "full", "stored_count": 3, "settings": {}})
        self.assertTrue(self.window.recovery_project_notice.isHidden())
