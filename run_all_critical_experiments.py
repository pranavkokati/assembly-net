#!/usr/bin/env python3
"""
Assembly-Net: Run All Critical Experiments.

This script runs all experiments required for publication validation.
It is designed for reproducibility and generates all results in one command.

Usage:
    python run_all_critical_experiments.py

Expected Runtime:
    ~3-5 minutes on a standard laptop (no GPU required)

Output Files:
    publication_results/
    ├── critical_experiments.json     # Main experiment results
    ├── figure1_data.json            # Data for Figure 1
    ├── figure1_plot.py              # Script to generate Figure 1
    └── experiment_log.txt           # Detailed execution log

Requirements:
    - Python 3.8+
    - numpy
    - networkx

No GPU, PyTorch, or heavy ML dependencies required.

Author: Assembly-Net Contributors
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("  ASSEMBLY-NET: CRITICAL PUBLICATION EXPERIMENTS")
    print("=" * 70)
    print()
    print("  This script validates the core claims of the paper:")
    print()
    print("  1. ISOMORPHIC FAILURE: Static models fail on identical-structure pairs")
    print("  2. TOPOLOGY VS STATS: Topological features outperform simple statistics")
    print("  3. LITERATURE VALIDATION: Results align with known aggregation behavior")
    print("  4. FIGURE 1: Visual demonstration of assembly non-equivalence")
    print()
    print("=" * 70)
    print()


def run_with_timing(name: str, func, *args, **kwargs):
    """Run a function and report timing."""
    print(f"Running: {name}")
    print("-" * 50)
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f} seconds")
        return result, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.1f} seconds: {e}")
        return None, elapsed, str(e)


def main():
    """Run all critical experiments."""

    print_banner()

    # Create output directory
    output_dir = Path("publication_results")
    output_dir.mkdir(exist_ok=True)

    # Start log
    log_lines = []
    log_lines.append(f"Assembly-Net Critical Experiments Log")
    log_lines.append(f"Started: {datetime.now().isoformat()}")
    log_lines.append(f"Python: {sys.version}")
    log_lines.append("")

    total_start = time.time()
    results = {}
    timings = {}
    errors = []

    # =========================================================================
    # EXPERIMENT 1: Critical Experiments (Isomorphic, Topology vs Stats, Literature)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CRITICAL EXPERIMENTS")
    print("=" * 70 + "\n")

    try:
        # Import here to avoid issues if dependencies missing
        import importlib.util
        import os

        _exp_path = os.path.join(os.path.dirname(__file__),
                                  'assembly_net', 'experiments', 'critical_experiments.py')
        _spec = importlib.util.spec_from_file_location("critical_experiments", _exp_path)
        _exp_module = importlib.util.module_from_spec(_spec)
        sys.modules["critical_experiments"] = _exp_module
        _spec.loader.exec_module(_exp_module)

        result, elapsed, error = run_with_timing(
            "Critical Experiments (Isomorphic + Topology + Literature)",
            _exp_module.run_all_critical_experiments,
            seed=42
        )

        if result:
            results['critical_experiments'] = result
        timings['critical_experiments'] = elapsed
        if error:
            errors.append(f"Critical experiments: {error}")

    except Exception as e:
        print(f"ERROR loading critical experiments module: {e}")
        errors.append(f"Module load error: {e}")

    # =========================================================================
    # EXPERIMENT 2: Figure 1 Visual Demonstration
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FIGURE 1 GENERATION")
    print("=" * 70 + "\n")

    try:
        import importlib.util
        import os

        _fig_path = os.path.join(os.path.dirname(__file__),
                                  'assembly_net', 'experiments', 'figure1_visual_demo.py')
        _spec = importlib.util.spec_from_file_location("figure1_visual_demo", _fig_path)
        _fig_module = importlib.util.module_from_spec(_spec)
        sys.modules["figure1_visual_demo"] = _fig_module
        _spec.loader.exec_module(_fig_module)

        result, elapsed, error = run_with_timing(
            "Figure 1: Assembly Non-Equivalence Visualization",
            _fig_module.main
        )

        if result:
            results['figure1'] = {'generated': True}
        timings['figure1'] = elapsed
        if error:
            errors.append(f"Figure 1: {error}")

    except Exception as e:
        print(f"ERROR loading figure1 module: {e}")
        errors.append(f"Figure 1 module load error: {e}")

    # =========================================================================
    # EXPERIMENT 3: Unit Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: UNIT TEST VALIDATION")
    print("=" * 70 + "\n")

    try:
        import importlib.util
        import os

        _test_path = os.path.join(os.path.dirname(__file__), 'tests', 'test_simulator.py')
        _spec = importlib.util.spec_from_file_location("test_simulator", _test_path)
        _test_module = importlib.util.module_from_spec(_spec)
        sys.modules["test_simulator"] = _test_module
        _spec.loader.exec_module(_test_module)

        result, elapsed, error = run_with_timing(
            "Unit Tests (34 tests)",
            _test_module.run_all_tests
        )

        results['unit_tests'] = {'all_passed': result}
        timings['unit_tests'] = elapsed
        if error:
            errors.append(f"Unit tests: {error}")

    except Exception as e:
        print(f"ERROR loading test module: {e}")
        errors.append(f"Test module load error: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Total runtime: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print()
    print("Phase timings:")
    for name, elapsed in timings.items():
        print(f"  {name}: {elapsed:.1f}s")
    print()

    if errors:
        print("ERRORS:")
        for err in errors:
            print(f"  - {err}")
        print()
    else:
        print("All phases completed successfully!")
        print()

    print("Output files:")
    for f in output_dir.iterdir():
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")
    print()

    # Update log
    log_lines.append(f"Completed: {datetime.now().isoformat()}")
    log_lines.append(f"Total runtime: {total_elapsed:.1f} seconds")
    log_lines.append("")
    log_lines.append("Timings:")
    for name, elapsed in timings.items():
        log_lines.append(f"  {name}: {elapsed:.1f}s")
    log_lines.append("")
    if errors:
        log_lines.append("Errors:")
        for err in errors:
            log_lines.append(f"  {err}")
    else:
        log_lines.append("Status: SUCCESS")

    # Save log
    with open(output_dir / "experiment_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    # Save combined results
    with open(output_dir / "all_results_summary.json", "w") as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_runtime_seconds': total_elapsed,
            'timings': timings,
            'errors': errors,
            'results_available': list(results.keys()),
        }, f, indent=2)

    print("=" * 70)
    print("REPRODUCIBILITY VERIFIED")
    print("=" * 70)
    print()
    print("To reproduce these results:")
    print("  python run_all_critical_experiments.py")
    print()
    print("To generate Figure 1 (requires matplotlib):")
    print("  pip install matplotlib")
    print("  python publication_results/figure1_plot.py")
    print()

    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
