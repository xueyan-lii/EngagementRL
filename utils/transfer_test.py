"""Sibling-problem selection for the transfer-test metric.

Rather than re-testing the student on the exact problem the tutor just
discussed (which any leak-adjacent tutoring can trivially inflate), we test
on a held-out "sibling" problem from the same domain and comparable base
difficulty, drawn from rd211/Big-Math-RL-Verified-Filtered's train split
(disjoint from the eval set, which is loaded from the test split). Matching
on domain alone is not reliable: of the dataset's 205 full-path domain tags,
many have only a single member, so a same-domain-only match would leave many
eval problems with zero candidates. Matching on domain AND llama8b_solve_rate
(a precomputed per-problem difficulty for the same probe-model class used
here) avoids handing out a systematically easier/harder sibling, which would
confound the transfer signal with a difficulty mismatch instead of isolating
the tutoring effect.
"""

import random
from collections import defaultdict
from typing import List, Optional, Tuple

from datasets import load_dataset

DEFAULT_TOLERANCES = (0.05, 0.10, 0.20)


def load_sibling_pool(name_or_path: str, exclude_problems: set):
    """Loads the train split as the sibling candidate pool, excluding any
    problem text that also appears in the eval set."""
    pool = load_dataset(name_or_path, split="train")
    return pool.filter(lambda r: r["problem"] not in exclude_problems)


def select_siblings(
    eval_problems: List[str],
    eval_domains: List[List[str]],
    eval_solve_rates: List[float],
    pool,
    seed: int,
    tolerances: Tuple[float, ...] = DEFAULT_TOLERANCES,
) -> Tuple[List[str], List[str], List[str]]:
    """For each eval problem, pick one sibling from `pool`, preferring a
    same-domain + similar-difficulty match, widening the tolerance and then
    dropping the domain constraint if nothing qualifies.

    Returns (sibling_problems, sibling_answers, fallback_log) -- fallback_log
    has one entry per eval problem describing which match tier was used, so
    callers can report how often the ideal (domain-matched) sibling wasn't
    available.
    """
    rng = random.Random(seed)

    pool_problems = pool["problem"]
    pool_answers = pool["answer"]
    pool_domains = pool["domain"]
    pool_solve_rates = pool["llama8b_solve_rate"]

    domain_index = defaultdict(list)
    for idx, doms in enumerate(pool_domains):
        for d in doms:
            domain_index[d].append(idx)

    sibling_problems, sibling_answers, fallback_log = [], [], []
    n_pool = len(pool_problems)

    for doms, solve_rate in zip(eval_domains, eval_solve_rates):
        candidates: List[int] = []
        tier: Optional[str] = None

        same_domain_idxs = set()
        for d in doms:
            same_domain_idxs.update(domain_index.get(d, []))

        for tol in tolerances:
            candidates = [
                idx
                for idx in same_domain_idxs
                if abs(pool_solve_rates[idx] - solve_rate) <= tol
            ]
            if candidates:
                tier = f"domain,tol={tol}"
                break

        if not candidates:
            for tol in tolerances + (1.0,):
                candidates = [
                    idx
                    for idx in range(n_pool)
                    if abs(pool_solve_rates[idx] - solve_rate) <= tol
                ]
                if candidates:
                    tier = f"no-domain,tol={tol}"
                    break

        chosen = rng.choice(candidates)
        sibling_problems.append(pool_problems[chosen])
        sibling_answers.append(pool_answers[chosen])
        fallback_log.append(tier)

    return sibling_problems, sibling_answers, fallback_log
