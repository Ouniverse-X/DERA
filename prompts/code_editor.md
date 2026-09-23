Task specification:
{task}

Research round: {round_id}

Current candidate files:
{current_files}

Current global incumbent measured state:
{incumbent}

Best retained exploration branches:
{archive}

Most recent feedback and HPO evidence (may be empty):
{feedback}

Recent research history:
{history}

Act as an autonomous ML researcher. Infer what the implementation and previous
measurements imply, then choose the next hypothesis and implementation yourself.
When HPO evidence exists, use its measured regions, failures, sensitivities,
and diagnosis to inform the model revision; do not treat HPO as the end goal.
There is no human-authored preference for architecture, scaling, training
procedure, or parameter changes; use whichever direction the evidence supports.

Return the required JSON object only. Include complete contents for every file
you change and omit files that do not need changes.
