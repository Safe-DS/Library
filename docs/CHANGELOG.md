## [0.6.0](https://github.com/Safe-DS/Stdlib/compare/v0.5.0...v0.6.0) (2023-03-27)


### Features

* allow calling `correlation_heatmap` with non-numerical columns ([#92](https://github.com/Safe-DS/Stdlib/issues/92)) ([b960214](https://github.com/Safe-DS/Stdlib/commit/b96021421f734fb7ca1b74e245a26b9997487292)), closes [#89](https://github.com/Safe-DS/Stdlib/issues/89)
* function to drop columns with non-numerical values from `Table` ([#96](https://github.com/Safe-DS/Stdlib/issues/96)) ([8f14d65](https://github.com/Safe-DS/Stdlib/commit/8f14d65611cd9a1d6c75ae2769a4e5551c42b766)), closes [#13](https://github.com/Safe-DS/Stdlib/issues/13)
* function to drop columns/rows with missing values ([#97](https://github.com/Safe-DS/Stdlib/issues/97)) ([05d771c](https://github.com/Safe-DS/Stdlib/commit/05d771c7fe9c0ea12ba7482a7ec5af197a450fce)), closes [#10](https://github.com/Safe-DS/Stdlib/issues/10)
* remove `list_columns_with_XY` methods from `Table` ([#100](https://github.com/Safe-DS/Stdlib/issues/100)) ([a0c56ad](https://github.com/Safe-DS/Stdlib/commit/a0c56ad1671bd4388356dd952b398efc31fd8796)), closes [#94](https://github.com/Safe-DS/Stdlib/issues/94)
* rename `keep_columns` to `keep_only_columns` ([#99](https://github.com/Safe-DS/Stdlib/issues/99)) ([de42169](https://github.com/Safe-DS/Stdlib/commit/de42169f6acde3d96985df24dc7f8213d96d2a4d))
* rename `remove_outliers` to `drop_rows_with_outliers` ([#95](https://github.com/Safe-DS/Stdlib/issues/95)) ([7bad2e3](https://github.com/Safe-DS/Stdlib/commit/7bad2e3e1b11fe45ed1fc408fa6289dfb5f301cb)), closes [#93](https://github.com/Safe-DS/Stdlib/issues/93)
* return new model when calling `fit` ([#91](https://github.com/Safe-DS/Stdlib/issues/91)) ([165c97c](https://github.com/Safe-DS/Stdlib/commit/165c97c107aa52fddb6951c7092f2dccb164c64d)), closes [#69](https://github.com/Safe-DS/Stdlib/issues/69)


### Bug Fixes

* handling of missing values when dropping rows with outliers ([#101](https://github.com/Safe-DS/Stdlib/issues/101)) ([0a5e853](https://github.com/Safe-DS/Stdlib/commit/0a5e853482ddeda147d5d6ff45e96166cfbfb1af)), closes [#7](https://github.com/Safe-DS/Stdlib/issues/7)

## [0.5.0](https://github.com/Safe-DS/Stdlib/compare/v0.4.0...v0.5.0) (2023-03-26)


### Features

* move plotting methods into `Column` and `Table` classes ([#88](https://github.com/Safe-DS/Stdlib/issues/88)) ([5ec6189](https://github.com/Safe-DS/Stdlib/commit/5ec6189a807092b00d38620403549c96a02164a5)), closes [#62](https://github.com/Safe-DS/Stdlib/issues/62)

## [0.4.0](https://github.com/Safe-DS/Stdlib/compare/v0.3.0...v0.4.0) (2023-03-26)


### Features

* better names for properties of `TaggedTable` ([#74](https://github.com/Safe-DS/Stdlib/issues/74)) ([fee398b](https://github.com/Safe-DS/Stdlib/commit/fee398b66cb9ae9e6675f455a8db31f271bfd207))
* change the name of a `Column` ([#76](https://github.com/Safe-DS/Stdlib/issues/76)) ([ec539eb](https://github.com/Safe-DS/Stdlib/commit/ec539eb6685d99183a35a138d1f345aaf6ae4c78))
* metrics as methods of models ([#77](https://github.com/Safe-DS/Stdlib/issues/77)) ([bc63693](https://github.com/Safe-DS/Stdlib/commit/bc636934a708b4a74aafed73fe4be75a7a36ebc4)), closes [#64](https://github.com/Safe-DS/Stdlib/issues/64)
* optionally pass type to column ([#79](https://github.com/Safe-DS/Stdlib/issues/79)) ([64aa429](https://github.com/Safe-DS/Stdlib/commit/64aa4293bdf035fe4f9a78b0b895c07f022ced3a)), closes [#78](https://github.com/Safe-DS/Stdlib/issues/78)
* remove `target_name` parameter of `predict` ([#70](https://github.com/Safe-DS/Stdlib/issues/70)) ([b513454](https://github.com/Safe-DS/Stdlib/commit/b513454c294f8ca03fbffa2b6f89a87e7d6fb9c6)), closes [#9](https://github.com/Safe-DS/Stdlib/issues/9)
* rename `tagged_table` parameter of `fit` to `training_set` ([#71](https://github.com/Safe-DS/Stdlib/issues/71)) ([8655521](https://github.com/Safe-DS/Stdlib/commit/8655521bebbca2da9c91e2db7a837d4869a1d527))
* return `TaggedTable` from `predict` ([#73](https://github.com/Safe-DS/Stdlib/issues/73)) ([5d5f5a6](https://github.com/Safe-DS/Stdlib/commit/5d5f5a69d7e4def34ab09494511ae6ad6a62d60b))

## [0.3.0](https://github.com/Safe-DS/Stdlib/compare/v0.2.0...v0.3.0) (2023-03-24)


### Features

* make `Column` and `Row` iterable ([#55](https://github.com/Safe-DS/Stdlib/issues/55)) ([74eea1f](https://github.com/Safe-DS/Stdlib/commit/74eea1f995d03732d14da16d4393e1d61ad33625)), closes [#47](https://github.com/Safe-DS/Stdlib/issues/47)


### Bug Fixes

* "UserWarning: X has feature names" when predicting ([#53](https://github.com/Safe-DS/Stdlib/issues/53)) ([74b0753](https://github.com/Safe-DS/Stdlib/commit/74b07536f418732025f10cd6dc048cb61fab6cc5)), closes [#51](https://github.com/Safe-DS/Stdlib/issues/51)
