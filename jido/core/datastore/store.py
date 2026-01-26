from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def write_run_outputs(
    out_dir: str | Path,
    machine_id: str,
    hardware: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir = Path(out_dir) / machine_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if hardware is not None:
        hardware_path = run_dir / "hardware.json"
        hardware_path.write_text(json.dumps(hardware, indent=2, sort_keys=True))

    if env is not None:
        env_path = run_dir / "env.json"
        env_path.write_text(json.dumps(env, indent=2, sort_keys=True))

    return run_dir
