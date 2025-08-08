from __future__ import annotations
from datetime import datetime, timezone
import re, random, string

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def kebab(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return re.sub(r'-+', '-', s).strip('-')

def short_id(n=4) -> str:
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))

def session_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{stamp}-{kebab(name)}-{short_id()}"
