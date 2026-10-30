"""Double-click launcher for the local customer interface."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parent
PORT=int(os.environ.get('FOODQUALITY_PORT','18501'))
if not 1024<=PORT<=65535:
    raise ValueError('Cổng phải nằm trong khoảng 1024–65535.')
URL=f'http://127.0.0.1:{PORT}'


def healthy():
    try:
        with urllib.request.urlopen(URL+'/_stcore/health',timeout=1) as response:
            return response.status==200
    except Exception:
        return False


def main():
    log_dir=ROOT/'workspace/user-app'
    log_dir.mkdir(parents=True,exist_ok=True)
    if not healthy():
        with (log_dir/'server.log').open('a',encoding='utf8') as log:
            process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-m','streamlit','run',str(ROOT/'user_app.py'),
                '--server.address','127.0.0.1','--server.port',str(PORT),'--server.headless','true',
                '--server.maxUploadSize','15','--browser.gatherUsageStats','false'],cwd=ROOT,stdout=log,stderr=log,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUTF8':'1'})
        for _ in range(40):
            if healthy():
                break
            if process.poll() is not None:
                raise RuntimeError('Khong khoi dong duoc. Xem workspace/user-app/server.log')
            time.sleep(.5)
        else:
            raise RuntimeError('Khoi dong cham. Xem workspace/user-app/server.log')
    webbrowser.open(URL)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if os.name=='nt':
            ctypes.windll.user32.MessageBoxW(None,str(exc),'FoodVision',0x10)
        sys.exit(1)
