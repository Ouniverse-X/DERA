Task specification:
{task}

Modified program:
{modified_program_files}

Code Agent's hypothesis:
{hypothesis}

Current best program and configuration:
{best}

Recent experiment history:
{history}

Initial evaluation of the modified program:
{initial_evaluation}

Total evaluation budget: {evaluation_budget}
Evaluations used, including the current Code Agent evaluation: {evaluations_used}
Evaluations remaining: {evaluations_remaining}
Maximum diagnostic trials in this iteration: {current_trial_limit}

Decide whether diagnostic trials are useful for the current program. If so, design a program-specific search space, select the TPE settings, and choose the number of trials within the stated limit. The trials should examine configurations whose results can help evaluate the modified program and guide the next code modification.

Return the required JSON object only.
