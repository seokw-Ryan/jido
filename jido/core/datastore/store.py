from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_run_outputs(
    out_dir: str | Path,
    machine_id: str,
    hardware: Dict[str, Any],
    env: Dict[str, Any],
) -> Path:
    run_dir = Path(out_dir) / machine_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hardware_path = run_dir / "hardware.json"
    env_path = run_dir / "env.json"

    hardware_path.write_text(json.dumps(hardware, indent=2, sort_keys=True))
    env_path.write_text(json.dumps(env, indent=2, sort_keys=True))

    return run_dir
