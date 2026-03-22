"""
Run all Memstate LangChain demos in sequence.
Each demo validates a different feature of the langchain-memstate package
against the live Memstate API.

Usage:
    python examples/run_all_demos.py

Optional env vars:
    MEMSTATE_API_KEY  - your Memstate API key (defaults to the test key)
"""
import sys
import traceback
import importlib.util
from pathlib import Path

DEMOS = [
    "demo_01_quickstart",
    "demo_02_versioning",
    "demo_03_time_travel",
    "demo_04_agent_tools",
    "demo_05_chat_history",
]

def run_demo(name: str) -> bool:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.main()
        return True
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("  Memstate LangChain Plugin -- End-to-End Demo Suite")
    print("=" * 60)
    print()

    results = {}
    for demo in DEMOS:
        print("-" * 60)
        passed = run_demo(demo)
        results[demo] = passed
        print()

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    all_passed = True
    for demo, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {demo}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All demos passed! Plugin is validated and ready.")
    else:
        print("  Some demos failed. See output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
