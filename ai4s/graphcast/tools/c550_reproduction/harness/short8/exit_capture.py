"""Capture diagnostics before the original hard exit, without changing training math."""
import os,sys,traceback

def install_exit_capture(capture):
    original=os._exit
    def wrapped(code):
        try:
            capture('runner_os_exit')
            sys.stdout.flush();sys.stderr.flush()
        except BaseException:
            traceback.print_exc();sys.stderr.flush()
            original(int(code) if int(code) else 1)
        original(int(code))
    os._exit=wrapped
    return original
