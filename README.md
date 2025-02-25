# Galerkin-Differencing
Work pertaining to getting AMR + side centered data working.

# running tests

this comes from the IBAMR repository
This is a bit ad-hoc at the moment. To run tests:

1. copy `attest.conf.in` to `attest.conf` in the top directory. This file
   configures `attest`, the test runner.
2. Edit `attest.conf` to contain the correct path to numdiff.
3. Run tests from the root directory via `./attest`.
4. Need to run "export PYTHONPATH=$(pwd)" for tests to be able to handle relative imports

# adding tests

Like IBAMR, each test consists of an input file, output file, and executable.
The executable should be a python script that has executable permissions set
(i.e., be sure to run `chmod +x`). Each executable should write a file named
"output" to its current working directory.

Tests pass if the output file they write matches (to some numerical tolerances
provided to `numdiff`) the output file saved in the test directory.
