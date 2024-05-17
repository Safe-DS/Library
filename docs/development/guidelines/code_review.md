# Code review

This document describes general guidelines for developers when reviewing a pull request (PR).

## Verify that the PR solves the addressed issue

### Understand the issue

* Read the issue that was addressed by the PR and make sure you understand it.
* Read any discussions that occurred on the issue page.

### Check the PR

* Check the PR for __completeness__: Was every point from the issue covered?
* Check for __correctness__: Were the problems described in the issue solved in the intended way?
* (For PRs that introduce new features:) Check the design - does it comply with these guidelines?
* Check for potential bugs - edge cases, exceptions that may be raised in functions that were called etc.
* Check the code style: Is it readable? Do variables have sensible names? Is it consistent to existing code? Is there a
  better / more elegant way to do certain things
  (e.g. [f-string](https://docs.python.org/3/tutorial/inputoutput.html#tut-f-strings) instead of manual string
  concatenation)?
* Check any issues reported by linters.

## Check the tests

* Run the tests locally.
* Check if there are any warnings that are not caught by the test suite.
* Make sure the test cases are complete - do they cover all edge cases (e.g. empty tables)?
* Make sure they comply to our [project guidelines][guidelines-tests] - are the tests parametrized and do all testcases
  have descriptive IDs?

## Verify that the PR does not break existing code

This is largely covered by the automated tests. However, you should always:

* Make sure all tests actually ran through (pytest, linter, code coverage).
* Make sure that the branch is up-to-date with the `main` branch, so you actually test the behaviour that will result
  once the feature branch is merged.

## Check the PR format

* Check that the PR title starts with a
  fitting [type](https://github.com/Safe-DS/.github/blob/main/.github/CONTRIBUTING.md#types)
  (e.g. `feat`, `fix`, `docs`, ...)
* Check that the changes introduced by the PR are documented in the PR message.

## Requesting changes

If you found any issues with the reviewed PR, navigate to the `Files changed` tab in the PR page and click on the
respective line to add your comments.

For more details on specific review features, check out the [documentation on GitHub][github-review].

## Finishing the review

When done, finish your review by navigating to the `Files changed` tab and clicking the green `Review changes` button on
the top right.

If you found no issues, select the `Approve` option to approve the changes made by the PR. If you found any problems,
select `Request Changes` instead.

[guidelines-tests]: tests.md

[github-review]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests
