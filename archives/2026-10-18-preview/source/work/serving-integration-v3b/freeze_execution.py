"""Freeze this unrun execution harness and both independently frozen plans."""
from pathlib import Path
from evidence import HERE, LAB, ROOT, freeze


def main():
    paths = [HERE/name for name in ("evidence.py","runtime.py","run_qualification.py","test_cpu.py","freeze_execution.py")]
    paths += [LAB/"qualified-profile.json", LAB/"provenance.json", LAB/"results/checkpoint-hashes.json"]
    paths += [ROOT/name for name in (
        "work/cone-experiment/cone.py", "work/cone-experiment/verify_real.py",
        "work/cone-v2/stabilization.py", "work/cone-dspark-stable/stabilization.py",
        "work/serving-cone-v3/cone_serving.py", "work/serving-cone-v3/manifest-01.json",
        "work/serving-cache-v3/serving_cache.py", "work/serving-cache-v3/hot_rows_upstream.py",
        "work/serving-cache-v3/UPSTREAM-LICENSE", "work/serving-cache-v3/provenance.json",
        "work/serving-cache-v3/manifest.json", "work/serving-cache-v3/NOTICE.txt",
        "work/serving-qualification-v3/plans/boundary-01.json",
        "work/serving-qualification-v3/plans/performance-01.json",
        "work/serving-qualification-v3/plans-manifest.json",
        "work/serving-qualification-v3/fixture-manifest.json",
        "work/serving-qualification-v3/source-manifest.json",
    )]
    record=freeze(HERE/"execution-manifest-01.json", paths,
        description="Unrun persistent-serving v3 boundary and diverse performance qualification. Native64 baseline is unchanged; stop after any retained matched-pair correctness/activation failure, never on speed alone.")
    print("Frozen",len(record["sources_sha256"]),"source/plan files")


if __name__ == "__main__":
    main()
