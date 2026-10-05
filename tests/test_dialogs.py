"""Format choices and session decisions must preserve their workflow meaning."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog

from kinect_scanner.gui.dialogs import ExportDialog, SessionProtectionDialog


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_format_mapping_and_texture_options(self):
        dialog = ExportDialog()
        self.assertEqual("glb", dialog.selected_format)
        dialog.texture_exposure_cb.setChecked(True)
        dialog.texture_best_cb.setChecked(True)
        for fmt in ("glb", "obj.zip", "ply", "obj"):
            dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(fmt))
            self.assertEqual(fmt, dialog.selected_format)
            textured = fmt in ("glb", "obj.zip")
            self.assertEqual(textured, dialog.texture_exposure_cb.isEnabled())
            self.assertEqual(textured, dialog.texture_best_cb.isEnabled())
            self.assertTrue(dialog.format_description.text())
        self.assertIn("do not support", dialog.format_description.text())
        dialog.format_combo.setCurrentIndex(0)
        self.assertEqual(
            {"exposure_correction": True, "blend_mode": "best"},
            dialog.texture_options,
        )
        dialog.texture_best_cb.setChecked(False)
        self.assertEqual("blend", dialog.texture_options["blend_mode"])
        dialog.close()

    def test_export_acceptance_only_returns_choice(self):
        dialog = ExportDialog()
        dialog.export_button.click()
        self.assertEqual(QDialog.DialogCode.Accepted, dialog.result())
        self.assertEqual("glb", dialog.selected_format)
        dialog.close()

    def test_session_choices_and_safe_default(self):
        for button_name, expected in (
            ("save_button", "save"),
            ("discard_button", "discard"),
            ("cancel_button", "cancel"),
        ):
            dialog = SessionProtectionDialog("new_scan")
            self.assertEqual("cancel", dialog.choice)
            self.assertTrue(dialog.save_button.isDefault())
            self.assertFalse(dialog.discard_button.isDefault())
            self.assertIn("starting a new scan", dialog.message_label.text())
            getattr(dialog, button_name).click()
            self.assertEqual(expected, dialog.choice)
            self.assertEqual(
                QDialog.DialogCode.Rejected if expected == "cancel" else QDialog.DialogCode.Accepted,
                dialog.result(),
            )

    def test_escape_and_window_close_cancel(self):
        dialog = SessionProtectionDialog("close")
        self.assertIn("closing the scanner", dialog.message_label.text())
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertEqual("cancel", dialog.choice)
        self.assertEqual(QDialog.DialogCode.Rejected, dialog.result())
        dialog = SessionProtectionDialog("close")
        dialog.show()
        dialog.close()
        self.assertEqual("cancel", dialog.choice)
        self.assertEqual(QDialog.DialogCode.Rejected, dialog.result())

    def test_enter_chooses_save(self):
        dialog = SessionProtectionDialog("new_scan")
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual("save", dialog.choice)
        self.assertEqual(QDialog.DialogCode.Accepted, dialog.result())

    def test_cancellation_prompt_names_the_action(self):
        dialog = SessionProtectionDialog("cancel_scan")
        self.assertIn("before cancelling the scan", dialog.message_label.text())
        self.assertEqual("cancel", dialog.choice)


if __name__ == "__main__":
    unittest.main()
