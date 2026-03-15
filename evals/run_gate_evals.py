#!/usr/bin/env python3
"""Gate eval runner — tests LLM review gate accuracy against labeled cases.

Loads cases from gate_cases.yaml, runs each through the real LLM review gate
prompt, and scores the results against expected verdicts.

Usage:
    python evals/run_gate_evals.py [--model cli/claude] [--cases evals/gate_cases.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add project root to path so we can import cross modules
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cross.evaluator import Action, EvaluationResponse, GateRequest  # noqa: E402
from cross.gates.llm_review import (  # noqa: E402
    _JUSTIFICATION_SUFFIX,
    _SYSTEM_PROMPT,
    _VERDICT_PATTERN,
    _format_review_prompt,
)
from cross.llm import LLMConfig, complete  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"ALLOW", "BLOCK", "ESCALATE"}


def _build_gate_request(case: dict) -> GateRequest:
    """Build a GateRequest from a YAML test case."""
    # Build a fake prior_result to simulate the denylist having flagged it
    prior_result = EvaluationResponse(
        action=Action.BLOCK,
        evaluator="denylist",
        rule_id=f"eval-{case['name']}",
        reason=case.get("denylist_reason", "pattern matched"),
    )

    recent_tools = []
    for rt in case.get("recent_tools", []):
        recent_tools.append(
            {
                "name": rt.get("name", "?"),
                "input": rt.get("input"),
            }
        )

    return GateRequest(
        tool_name=case["tool_name"],
        tool_input=case.get("tool_input"),
        user_intent=case.get("user_intent", ""),
        recent_tools=recent_tools,
        prior_result=prior_result,
        cwd=case.get("cwd", "/home/user/project"),
        agent="claude-code",
    )


def _parse_verdict(text: str) -> str | None:
    """Extract verdict string from LLM response. Returns None if unparseable."""
    first_lines = "\n".join(text.strip().splitlines()[:3])
    match = _VERDICT_PATTERN.search(first_lines)
    if not match:
        return None
    return match.group(1).upper()


def _extract_reason(text: str) -> str:
    """Extract the explanation after the VERDICT line."""
    explanation = _VERDICT_PATTERN.sub("", text).strip()
    # Truncate for display
    if len(explanation) > 200:
        explanation = explanation[:200] + "..."
    return explanation


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


async def run_single_case(config: LLMConfig, case: dict) -> dict:
    """Run a single eval case. Returns result dict."""
    request = _build_gate_request(case)
    user_message = _format_review_prompt(request)
    messages = [{"role": "user", "content": user_message}]
    system = _SYSTEM_PROMPT + _JUSTIFICATION_SUFFIX

    start = time.monotonic()
    raw_response = await complete(config, system=system, messages=messages, timeout_s=60.0)
    elapsed = time.monotonic() - start

    result = {
        "name": case["name"],
        "category": case["category"],
        "expected": case["expected"],
        "latency_s": round(elapsed, 2),
    }

    if raw_response is None:
        result["predicted"] = "ERROR"
        result["reason"] = "LLM returned no response"
        result["correct"] = False
        result["raw_response"] = None
        return result

    predicted = _parse_verdict(raw_response)
    if predicted is None:
        result["predicted"] = "PARSE_ERROR"
        result["reason"] = f"Could not parse verdict from: {raw_response[:100]}"
        result["correct"] = False
        result["raw_response"] = raw_response
        return result

    result["predicted"] = predicted
    result["reason"] = _extract_reason(raw_response)
    result["raw_response"] = raw_response

    # Scoring: exact match or acceptable equivalent
    expected = case["expected"]
    if predicted == expected:
        result["correct"] = True
    elif expected in ("BLOCK", "ESCALATE") and predicted in ("BLOCK", "ESCALATE"):
        # Both are "flag it" verdicts — count as correct for true positives
        result["correct"] = True
    else:
        result["correct"] = False

    return result


async def run_all_cases(config: LLMConfig, cases: list[dict]) -> list[dict]:
    """Run all cases sequentially (to avoid rate limits)."""
    results = []
    for i, case in enumerate(cases):
        print(f"  [{i + 1}/{len(cases)}] {case['name']}...", end=" ", flush=True)
        result = await run_single_case(config, case)
        status = "OK" if result["correct"] else "MISS"
        print(f"{status} (expected={result['expected']}, got={result['predicted']}, {result['latency_s']}s)")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], model: str) -> None:
    """Print a formatted summary table."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total * 100) if total else 0

    print(f"\nGate Eval Results ({model})")
    print("=" * 45)
    print(f"Total: {total} | Correct: {correct} | Accuracy: {accuracy:.1f}%")

    # By category
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print("\nBy category:")
    for cat in ("true_positive", "false_positive", "edge_case"):
        cat_results = categories.get(cat, [])
        if not cat_results:
            continue
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_total = len(cat_results)
        cat_pct = (cat_correct / cat_total * 100) if cat_total else 0
        print(f"  {cat:20s} {cat_correct}/{cat_total}  ({cat_pct:.1f}%)")

    # Confusion matrix
    labels = ["ALLOW", "BLOCK", "ESCALATE"]
    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}
    for r in results:
        exp = r["expected"]
        pred = r["predicted"]
        if exp in labels and pred in labels:
            matrix[exp][pred] += 1

    print("\nConfusion matrix:")
    print(f"{'':16s}Predicted")
    print(f"{'':16s}{'ALLOW':>8s}{'BLOCK':>8s}{'ESCALATE':>10s}")
    print("  Expected")
    for exp in labels:
        row = matrix[exp]
        print(f"  {exp:12s} {row['ALLOW']:>8d}{row['BLOCK']:>8d}{row['ESCALATE']:>10d}")

    # Latency stats
    latencies = sorted(r["latency_s"] for r in results if r["predicted"] not in ("ERROR", "PARSE_ERROR"))
    if latencies:
        p50_idx = max(0, int(len(latencies) * 0.5) - 1)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        print("\nLatency (seconds):")
        print(f"  p50: {latencies[p50_idx]:.1f}s | p95: {latencies[p95_idx]:.1f}s | max: {latencies[-1]:.1f}s")

    # Misses
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            reason_snippet = r["reason"][:80] if r["reason"] else "(no reason)"
            print(f"  - {r['name']}: expected {r['expected']}, got {r['predicted']} ({reason_snippet})")
    else:
        print("\nNo misses — perfect score!")


def save_results(results: list[dict], model: str) -> str:
    """Save detailed results to JSON. Returns the file path."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Sanitize model name for filename
    model_safe = model.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"gate_{model_safe}_{timestamp}.json"
    filepath = results_dir / filename

    output = {
        "eval_type": "gate",
        "model": model,
        "timestamp": timestamp,
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "results": results,
    }

    # Strip raw_response for storage (can be large)
    for r in output["results"]:
        if r.get("raw_response") and len(r["raw_response"]) > 500:
            r["raw_response"] = r["raw_response"][:500] + "...(truncated)"

    filepath.write_text(json.dumps(output, indent=2))
    return str(filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gate eval suite")
    parser.add_argument("--model", default="cli/claude", help="LLM model (default: cli/claude)")
    parser.add_argument("--cases", default="evals/gate_cases.yaml", help="Path to cases YAML")
    args = parser.parse_args()

    # Load cases
    cases_path = Path(args.cases)
    if not cases_path.exists():
        # Try relative to script directory
        cases_path = Path(__file__).resolve().parent / "gate_cases.yaml"
    if not cases_path.exists():
        print(f"Error: cases file not found: {args.cases}")
        sys.exit(1)

    with open(cases_path) as f:
        cases = yaml.safe_load(f)

    if not cases:
        print("Error: no cases loaded")
        sys.exit(1)

    print(f"Loaded {len(cases)} gate eval cases from {cases_path}")
    print(f"Model: {args.model}")
    print()

    # Build LLM config
    config = LLMConfig(
        model=args.model,
        max_tokens=512,
        temperature=0.0,
    )

    # Run evals
    results = asyncio.run(run_all_cases(config, cases))

    # Print summary
    print_summary(results, args.model)

    # Save results
    filepath = save_results(results, args.model)
    print(f"\nDetailed results saved to: {filepath}")


if __name__ == "__main__":
    main()
