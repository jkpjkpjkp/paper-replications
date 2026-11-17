# Paper references

Use `PAPER_TEXT_INDEX.txt` to locate searchable paper references. Top-level project
folders keep `PAPER_*` files in `.agent/`; nested projects retain their existing
locations. The index lists each reference directory. `PAPER_TEXT.txt` and
`PAPER_INDEX.json` record source hashes, coverage, page and line mappings. PDF
extracts also have `PAPER_LAYOUT.txt` for table layout. Extract source paths and
index `path_relative_to_project` values remain relative to the project root.
Use `rg --hidden` when searching `.agent/` directories.
Check the coverage field: an abstract, linked guide or study overview is not a
full paper. Verify equations, tables and figures against the original source.
`_replication/prepare_paper_texts.py` rebuilds these from the curated source map;
it does not modify archived sources or replication assessments.

# Replication status updates

Whenever completed evidence changes a project's replication assessment, update
that project's `STATUS.txt` in the same task. Apply the grading rules in
`/home/jkp/.codex/skills/replication-status/SKILL.md`; reassess the first-line
integer grade and record the assessment date, experiment scope, paper and local
metrics, error/tolerance, evidence paths, and remaining gaps. Refresh the evidence
even when the integer grade stays the same.

Keep `/home/jkp/data/REPLICATION_STATUS.txt` synchronized: update the project's
entry, comparison summary, grade grouping, and aggregate counts when applicable.
Preserve earlier failures and unresolved mismatches prominently. If a successful
baseline or released-artifact analysis supports a scoped level 2, state that scope
explicitly; do not imply that the main method or the whole paper was reproduced.
Update conflicting current-grade statements in project reports as well.
