"""Desktop entry point: retain errors even when launched without a console."""
from datetime import datetime
from pathlib import Path
import os
import sys
import traceback


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    log_path = root / 'workspace/gui-runtime/startup.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log
        try:
            print(f'\n[{datetime.now().isoformat()}] Starting FoodQuality GUI')
            from gui import App
            app = App()
            if '--review' in sys.argv:
                app.after(200, app._open_review)

            def report_callback_error(exc_type, exc_value, tb):
                traceback.print_exception(exc_type, exc_value, tb, file=log)
                from tkinter import messagebox
                messagebox.showerror('FoodQuality - Loi', f'{exc_value}\n\nChi tiet: {log_path}', parent=app)

            app.report_callback_exception = report_callback_error
            app.update_idletasks()
            print('GUI ready')
            app.mainloop()
            return 0
        except Exception:
            traceback.print_exc(file=log)
            log.flush()
            if os.name == 'nt':
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f'Khong mo duoc FoodQuality.\nChi tiet loi:\n{log_path}', 'FoodQuality - Loi khoi dong', 0x10)
            return 1
        finally:
            sys.stdout, sys.stderr = previous_stdout, previous_stderr


if __name__ == '__main__':
    raise SystemExit(main())
