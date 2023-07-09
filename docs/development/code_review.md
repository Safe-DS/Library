# Code review

This document describes
general guidelines for reviewers
when reviewing a Pull Request (PR).

## Verify that the PR solves the addressed issue

### Understand the issue

* Read the issue that was addressed by the PR and make sure you understand it
* Read any discussions that occurred on the issue page

### Check the Pull Request

* Check the Pull Request for __completeness__: Was every point from the issue covered?
* Check for __correctness__: Were the problems described in the issue solved in the intended way?
* (For PRs that introduce new features:) Check the design - does it comply with our [`guidelines`][guidelines-general]?
* Check for potential bugs - edge cases, exceptions that may be raised in functions that were called etc.
* Check the code style: Is it readable? Do variables have sensible names? Is it consistent to existing code? Is there a better / more elegant way to do certain things (e.g. f-string instead of manual string concatenation)?
* Check any warnings reported by your IDE

## Check the tests

* Run the tests locally
* Check if there are any warnings that are not caught by the test suite
* Make sure the test cases are complete - do they cover all edge cases (e.g. empty tables)?
* Make sure they comply to our [`guidelines`][guidelines-tests] - are the tests parametrized and do all testcases have descriptive IDs?

## Verify that the PR does not break existing code

This is largely covered by the automated tests.
However, you should always:

* Make sure all tests actually ran through (pytest, linter, code coverage)
* Make sure that the branch is up-to-date with the `main` branch, so you actually test the behaviour that will result once the feature branch is merged

## Check the Pull Request format

* Check that the PR title starts with a fitting [type](https://github.com/Safe-DS/.github/blob/main/.github/CONTRIBUTING.md#types) (e.g. `feat`, `fix`, ...)
* Check that the changes introduced by the PR are documented in the PR message
* Check that the `time spent by team` has been updated on the issue page

[guidelines-general]: https://stdlib.safeds.com/en/stable/development/guidelines/
[guidelines-tests]: https://stdlib.safeds.com/en/stable/development/guidelines/#tests
