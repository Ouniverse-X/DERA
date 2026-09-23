Task specification:
{task}

Candidate code change:
{change_summary}

Current global incumbent (a jointly measured frozen code/configuration):
{incumbent}

Candidate default-configuration development objective:
{default_objective}

Candidate's best actually measured frozen state:
{candidate_best}

Pilot HPO summary (all observations are from development/validation):
{hpo_summary}

Recommend whether this branch should be kept, discarded, or debugged, diagnose
what the probe revealed, and name the next research focus. The controller will
independently promote any measured candidate that beats the incumbent. The
pilot is not an invitation to increase HPO budget. Return the required JSON
object only.
