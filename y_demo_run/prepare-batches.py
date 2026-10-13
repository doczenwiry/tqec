import json
from pathlib import Path
from tqec.orchestration import BatchConfig, prepare_batch

config = BatchConfig.from_dict(json.loads(Path("config.json").read_text()))
gadgets = sorted(Path("gadgets").glob("*.bgraph"))
manifest = prepare_batch(gadgets, config, "batch")
for unit in manifest.units:
    print(f"{unit.name:26} {unit.status:15} k={sorted(unit.circuits)}")