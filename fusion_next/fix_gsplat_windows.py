"""Local gsplat 1.5.3 JIT compatibility fix: MSVC flags, not rasterizer logic."""
import importlib.util
from pathlib import Path
import os

if os.name != 'nt':
    raise SystemExit('This compatibility fix is Windows-only')
root = Path(importlib.util.find_spec('gsplat').origin).parent
path = root/'cuda/_backend.py'
old = 'extra_cflags = [opt_level, "-Wno-attributes"]'
new = 'extra_cflags = (["/Od" if FAST_COMPILE else "/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"])'
text = path.read_text()
if new in text:
    print('MSVC compatibility already applied')
elif text.count(old) == 1:
    backup = path.with_suffix('.py.original')
    if not backup.exists(): backup.write_text(text)
    path.write_text(text.replace(old, new))
    print('Applied MSVC compile-flag compatibility:', path)
else:
    raise SystemExit('Unexpected gsplat version; refusing to patch')
