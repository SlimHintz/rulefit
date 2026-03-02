"""Run RuleFit tree-generation benchmarks from the command line."""

from __future__ import annotations

import argparse

from rulefit.benchmark import benchmark_all, format_results_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RuleFit tree generators")
    parser.add_argument("--reg-samples", type=int, default=5000)
    parser.add_argument("--reg-features", type=int, default=25)
    parser.add_argument("--clf-samples", type=int, default=5000)
    parser.add_argument("--clf-features", type=int, default=25)
    parser.add_argument("--reg-max-rules", type=int, default=1000)
    parser.add_argument("--clf-max-rules", type=int, default=1000)
    parser.add_argument("--reg-tree-size", type=int, default=5)
    parser.add_argument("--clf-tree-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = benchmark_all(
        reg_n_samples=args.reg_samples,
        reg_n_features=args.reg_features,
        reg_max_rules=args.reg_max_rules,
        reg_tree_size=args.reg_tree_size,
        reg_random_state=args.seed,
        clf_n_samples=args.clf_samples,
        clf_n_features=args.clf_features,
        clf_max_rules=args.clf_max_rules,
        clf_tree_size=args.clf_tree_size,
        clf_random_state=args.seed,
    )
    print(format_results_table(results))


if __name__ == "__main__":
    main()
