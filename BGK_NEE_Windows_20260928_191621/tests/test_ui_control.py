"""Qt signal/slot integration on disposable processes; run with offscreen Qt."""
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dashboard'))
import unittest
from pathlib import Path
import test_control as fixtures
from dashboard import Dashboard, W, QtGui, FONT_DIR
from control import snapshot


class QtControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = W.QApplication.instance() or W.QApplication([])
        cls.app.setStyle('Fusion')
        for font in FONT_DIR.glob('*-Regular.ttf'):
            QtGui.QFontDatabase.addApplicationFont(str(font))

    def wait_until(self, predicate):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate(): return
            time.sleep(.01)
        self.fail('Qt condition timed out')

    def test_worker_controls_and_cancel_confirmation(self):
        fixture = fixtures.ProcessControlTests(); fixture.setUp()
        window = Dashboard(fixture.root)
        try:
            self.wait_until(lambda: window.data is not None)
            self.assertTrue(window.control_buttons['pause'].isEnabled())
            self.assertFalse(window.control_buttons['resume'].isEnabled())
            window.request_control('pause')
            self.wait_until(lambda: not window.control_pending and window.data['controls'].get('child_paused'))
            self.assertFalse(window.control_buttons['pause'].isEnabled())
            self.assertTrue(window.control_buttons['resume'].isEnabled())
            self.assertIn('已暂停', window.status.text())
            window.request_control('resume')
            self.wait_until(lambda: not window.control_pending and not window.data['controls'].get('child_paused'))
            self.assertTrue(window.control_buttons['pause'].isEnabled())
            window.request_control('terminate')
            self.assertIsNotNone(window.confirmation)
            window.confirmation.reject()
            self.app.processEvents()
            self.assertTrue(snapshot(fixture.root)['available'])
            self.assertFalse((fixture.root/'STOP').exists())
            old = fixture.root/'calculation/cavity_old'; old.mkdir()
            (old/'run_request.json').write_text('{"config":{}}')
            (old/'progress.jsonl').write_text('{"iteration":1}\n')
            window.refresh()
            self.wait_until(lambda: window.selector.findData(str(old)) >= 0)
            window.selector.setCurrentIndex(window.selector.findData(str(old)))
            self.assertFalse(any(b.isEnabled() for b in window.control_buttons.values()))
            self.wait_until(lambda: window.data['folder'] == str(old))
            self.assertFalse(any(b.isEnabled() for b in window.control_buttons.values()))
        finally:
            window.close(); fixture.tearDown()

if __name__ == '__main__': unittest.main()
