# Validation record

Validated on November 17, 2026. This record covers the standalone Table 1 package,
separate from the legacy local evaluation and upstream reproduction runs.

- Source: authors' upstream commit `bde5d07442fdea2c230641ebfbdd0ea00b5021fc`.
- Targets: 48 numeric cells, manually transcribed and visually checked against
  Table 1, page 11, January 25, 2026 SSRN preprint.
- Local input run: **48/48 matches** at three decimals; 24 series, each containing
  all 288 months from 2001-01 through 2024-12.
- Offline unit suite: **14 tests passed**, including a separate direct-formula
  check of Bartlett HAC variance and input/acquisition failure paths.
- Fresh-download run: both CSVs fetched from the pinned GitHub URLs, both hashes
  verified, **48/48 matches**. Cached offline rerun produced an identical
  `table1.csv`.
- Clean source archive: extracted into a separate temporary directory, installed
  with `uv sync --frozen --python 3.11`, passed all **14 tests**, fetched inputs
  independently, and obtained **48/48 matches**. Its offline rerun also produced
  an identical `table1.csv`.
- Python versions tested: **3.11.16** in the extracted release and **3.12.13** in
  a newly created project environment. Both used the checked-in `uv.lock`.
- Wheel built successfully and inspected for packaged input hashes, expected
  values, and upstream license notice.
- Release audit: every member's SHA-256 verified against the included manifest;
  no raw data, downloaded CSVs, PDFs, upstream checkout, model weights, local
  results, legacy scripts, or virtual environments in the archive.
- Original README preserved byte-for-byte as `README-evaluation.txt`, SHA-256
  `46f4dc62c126fd0aa6ed492fea8f1ad004f732700303df398d8b115b9a0e3460`.

CI is configured for Python 3.11 and 3.12. The above results are local execution
evidence; no hosted CI run or public repository publication was performed.

Re-run the commands in the README to obtain current, machine-specific manifests.
Generated result files are intentionally excluded from the source release.
