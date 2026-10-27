# Ground rules for this repo

1. Never write a metric, benchmark or percentage into a README, doc or commit message unless a script in this repo produced it in this session. Show the command and its raw output first.
2. Every reported number needs: the command that produced it, the dataset or split, N, the hardware, and the date.
3. If a result is worse than expected, report it as-is. Do not tune on the test set to make it look better.
4. Do not rewrite git history or force-push without asking first.
5. Do not add a dependency, framework or tool to the README unless it is imported and used in the code.
6. Before finishing, list every claim added to the README and the file and line that backs it.
