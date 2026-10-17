import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dataset_export_gui import ExportWindow


class ExportWindowTests(unittest.TestCase):
    def test_review_and_export_callbacks_from_tab_and_separate_window(self):
        root=tk.Tk()
        root.withdraw()
        try:
            root._open_review=Mock()
            release=tk.StringVar(root)
            root._model_tabs=[SimpleNamespace(release=release)]
            for parent in (tk.Frame(root),tk.Toplevel(root)):
                with self.subTest(parent=type(parent).__name__), patch.object(ExportWindow,'refresh'):
                    window=ExportWindow(parent,Path.cwd())
                    window.withdraw()
                    try:
                        self.assertIs(window.app,root)
                        before=root._open_review.call_count
                        window.open_review()
                        self.assertEqual(root._open_review.call_count,before+1)
                        window.events.put(('ok','fixture-release'))
                        window.poll()
                        self.assertEqual(release.get(),'fixture-release')
                    finally:
                        window.destroy()
                parent.destroy()
        finally:
            root.destroy()
