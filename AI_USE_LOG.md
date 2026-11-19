# AI assistance record

Date: 11 November 2026. Tool: ChatGPT.
Assistance: topic refinement; source discovery; proposed experimental protocol; Python implementation;
Colab notebook; documentation; synthetic software checks.
Empirical UCI results: primary run and duplicate-removal sensitivity executed. Metrics and plots
were produced by code and independently checked against saved predictions. AI also drafted the
interpretation in FINDINGS.md. This does not make the package team-authored or meet the 20% limit.
Team-authored work at delivery: none recorded. Do not misrepresent this package as team-written.

The assignment specifies below 20% AI-generated text/code and an explicit declaration. This scaffold
is substantially AI-generated. It is not certified to meet that requirement. Rewording or cosmetic
code edits do not reliably establish a compliant percentage. Confirm what assistance is permitted,
and record genuine team implementation, decisions, validation, interpretation and writing.

Maintain columns Date / Person / Assistance / Independently completed work / Evidence / Final use.
Do not use paraphrasing or detector tricks to disguise AI assistance.

## 19 November 2026: Claude Code (Anthropic)
Assistance: repository audit against the AMLBT template; unpacked the delivery bundle into the repository;
added the `--dedup` command-line option to experiment.py; wrote extended_analysis.py (post-hoc ablation,
rule baselines, cutoff bootstrap, seed and tuning checks, recalibration, drift, permutation importance,
rolling cutoff, capacity curve, cost scenarios); wrote tests/; pinned the notebook install; drafted
EXTENDED_FINDINGS.md, README.md updates, VALIDATION.md addendum and the IEEE LaTeX draft in report/.
Verification performed by the tool: reran the primary and sensitivity pipelines and compared every
published CSV (max difference 1e-12); 12 automated tests pass.
Team-authored work in this session: none. The AI share of code and text increased in this session.
Later the same day: Claude Code wrote report/make_figures.py and replaced the report's seven single plots
with six multi-panel figures (16 panels), and added team names from the course team sheet to the author block.
Later the same day: at the user's request, Claude Code ran a humanizer editing pass over report/seasonsafe.tex
(plainer wording, no en/em dashes in prose; all numbers, citations and claims unchanged). This changes
phrasing only; it does not change who wrote the text or the AI share.
