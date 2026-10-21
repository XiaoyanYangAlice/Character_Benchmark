"""手动测试跑分器：很多环境（包括集群登录节点）没装 pytest，用这个脚本代替——
import 每个 tests.test_* 模块，把里面所有 test_ 开头的函数都跑一遍，汇总
通过/失败数。放在仓库根目录，跑法：

    cd <仓库根目录>
    python run_tests.py
"""
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tests  # noqa: E402  (需要先插入 sys.path 才能找到 tests 包)

total = 0
failed = 0

for mod_info in pkgutil.iter_modules(tests.__path__, prefix="tests."):
    modname = mod_info.name
    if not modname.rsplit(".", 1)[-1].startswith("test_"):
        continue
    try:
        mod = importlib.import_module(modname)
    except Exception:
        print(f"IMPORT FAILED: {modname}")
        traceback.print_exc()
        continue
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                print(f"PASS {modname}.{name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {modname}.{name}: {e}")
                traceback.print_exc()

print(f"\n{total} tests, {failed} failed")
sys.exit(1 if failed else 0)
