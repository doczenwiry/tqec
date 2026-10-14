import stim
from tqec.orchestration import BatchManifest, BatchResult
from tqec.utils.noise_model import NoiseModel

if __name__ == "__main__":
    run = "batch"
    manifest = BatchManifest.read(run)
    errors = {(r.gadget_id, r.k, r.p): f"{r.errors}/{r.shots}" for r in BatchResult.read(run).results}
    print(f"{'gadget':26} {'status':8} k  dist 2k+1  p      errors")
    for unit in manifest.units:
        if not unit.circuits:
            print(f"{unit.name:26} {unit.status}")
        for k, rel in sorted(unit.circuits.items()):
            circuit = stim.Circuit.from_file(manifest.run_dir / rel)
            noisy = NoiseModel.uniform_depolarizing(1e-3).noisy_circuit(circuit)
            error = noisy.shortest_graphlike_error(
                ignore_ungraphlike_errors=False, canonicalize_circuit_errors=True
            )
            for p in manifest.config.ps:
                count = errors.get((unit.gadget_id, k, p), "-")
                print(f"{unit.name:26} {unit.status:8} {k}  {len(error):>4} {2 * k + 1:>4}  {p:<6g} {count}")