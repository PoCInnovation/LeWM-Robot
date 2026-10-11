import re

files = [
    "source/sim_to_real_so101/tasks/so101_env_cfg.py",
    "source/sim_to_real_so101/tasks/task_env_cfg.py",
]

for f in files:
    text = open(f, encoding="utf-8").read()
    print("=" * 20, f, "=" * 20)
    m = re.search(r"class \w*Actions\w*Cfg.*?(?=\nclass |\Z)", text, re.S)
    print(m.group(0) if m else "[no ActionsCfg class found]")