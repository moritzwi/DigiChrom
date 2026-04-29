"""PyInstaller entry point — starts the Streamlit server and opens the browser."""

import sys
import threading
import time
import webbrowser
from pathlib import Path


def _app_path() -> str:
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
    return str(base / "app.py")


def _open_browser(port: int = 8501, delay: float = 3.0) -> None:
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", _app_path(),
        "--server.headless=true",
        "--server.port=8501",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
    ]
    sys.exit(stcli.main())
