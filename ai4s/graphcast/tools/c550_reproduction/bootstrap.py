"""Use the existing interpreter's stdlib first; append explicit installed paths."""
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
settings = json.loads((root / 'job.json').read_text())['settings']
paths = [str(root), settings['flaggems_root'] + '/src', settings['model_root'] + '/src',
         *settings['dependency_paths'], *settings['python_site_paths']]
for path in paths:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    if path not in sys.path:
        sys.path.append(path)
sys.argv = sys.argv[1:]
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('assertion-eliding Python modes are forbidden')
if sys.argv[0] == '--module':
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
else:
    runpy.run_path(sys.argv[0], run_name='__main__')
