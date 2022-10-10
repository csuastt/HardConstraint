"""
Wrap a function to suppress its stdoutput.
Source: anonymous
"""
import contextlib
import sys

class DummyFile(object):
    def write(self, _): pass

    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
