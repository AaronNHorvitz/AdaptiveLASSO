# Notice

This repository implements the adaptive LASSO methodology for linear
regression.

## Academic provenance

The adaptive LASSO method is based on:

- Hui Zou, *The Adaptive Lasso and Its Oracle Properties*, Journal of the
  American Statistical Association, 101(476), 1418-1429, 2006.
- DOI: `10.1198/016214506000000735`

## Repository provenance

An earlier exploratory version of this repository was informed by Alexandre
Gramfort's public adaptive-LASSO demonstration gist:

- https://gist.github.com/agramfort/1610922

That gist indicates a BSD 3-clause license.

The current repository has since been rewritten into a Statsmodels-style Python
package with:

- an `OLS`-compatible model interface
- packaging metadata
- automated tests
- documentation
- an example notebook

This notice is included to preserve attribution and implementation history. It
is not itself a substitute for a repository license.
