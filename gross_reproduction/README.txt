ZERO-COST ANNOUNCEMENT-TIMING REPRODUCTION

From /home/jkp/data/earnings-timing run:
  .venv/bin/python gross_reproduction/reproduce.py

Read results/report.html or results/summary.txt.

This is an independent reconstruction of five existing timing-only daily
portfolios, with no new search. It uses raw CRSP daily total returns, rebuilds
the median forecast clock from prior available announcements, and retains the
previously validated regular-65 / five-analyst universe. Every reconstructed
daily portfolio is compared with its prior saved gross return.

All costs, including short borrow, equal zero. Both long and short legs equal
100% of NAV; total gross exposure is 200%. Arithmetic daily means include cash
days. At 100% total gross exposure the reported daily means halve.

The median-window rule highlighted here was selected within its family using
earlier development data, but is highlighted in this report after prior gross
results were inspected. This is retrospective reproduction, not a new untouched
test or an exact replication of an identified external paper.
