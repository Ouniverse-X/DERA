Task specification:
{task}

Candidate files after the research edit:
{candidate_files}

Code Agent's hypothesis:
{change_summary}

Current global incumbent:
{incumbent}

Retained exploration branches:
{archive}

Recent measured history:
{history}

The candidate's just-completed default evaluation:
{default_evaluation}

Global budget for extra HPO trials only: {total_hpo_budget}
Extra HPO trials already used: {hpo_trials_used}
Extra HPO trials still available: {remaining_hpo_budget}
Maximum extra HPO trials allowed in this round: {per_round_hpo_limit}
Research rounds remaining including this one: {rounds_remaining}

The default result above has already been measured and costs no HPO budget.
Autonomously decide whether TPE is an efficient way to answer a concrete model
research question now, what to search, its TPE settings, and how many of the
scarce extra trials to allocate. Preserve budget when HPO would not provide
actionable evidence for a later model revision.
Choose the actual `tpe_trials` autonomously from zero through the stated
per-round limit. The limit is a resource constraint, not a request
to use all of it.
Return JSON only.
