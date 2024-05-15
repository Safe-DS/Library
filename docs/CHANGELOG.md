## [0.25.0](https://github.com/Safe-DS/Library/compare/v0.24.0...v0.25.0) (2024-05-15)


### Features

* major API redesign (WIP) ([#752](https://github.com/Safe-DS/Library/issues/752)) ([8e781f9](https://github.com/Safe-DS/Library/commit/8e781f9728c09b8f94fcb22294bca41eea269d8a)), closes [#694](https://github.com/Safe-DS/Library/issues/694) [#699](https://github.com/Safe-DS/Library/issues/699) [#714](https://github.com/Safe-DS/Library/issues/714) [#748](https://github.com/Safe-DS/Library/issues/748)
* move NN converters and layers to separate packages ([#759](https://github.com/Safe-DS/Library/issues/759)) ([c6a4073](https://github.com/Safe-DS/Library/commit/c6a40731094cc8649280843063ca0a8cf51ac256))
* remove operations without replacement from tabular containers ([#747](https://github.com/Safe-DS/Library/issues/747)) ([0e5a54b](https://github.com/Safe-DS/Library/commit/0e5a54b0e02d3cce427728f8adf3717176340b8b))
* specify partial order in label encoder ([#763](https://github.com/Safe-DS/Library/issues/763)) ([6fbe537](https://github.com/Safe-DS/Library/commit/6fbe5379ae7676906bf8d0f21fe00926d792c60f)), closes [#639](https://github.com/Safe-DS/Library/issues/639)


### Bug Fixes

* Conversion of tabular dataset to tensors ([#757](https://github.com/Safe-DS/Library/issues/757)) ([9e40b65](https://github.com/Safe-DS/Library/commit/9e40b65e9f18f7af3a376fbc2e8ee0b5435541aa))
* fixed devices with new polars implementation ([#756](https://github.com/Safe-DS/Library/issues/756)) ([e72339e](https://github.com/Safe-DS/Library/commit/e72339e370836bdd285a4746a203f4e91b10436c))


### Performance Improvements

* implement one hot encoder and imputer using polars ([#768](https://github.com/Safe-DS/Library/issues/768)) ([e993c17](https://github.com/Safe-DS/Library/commit/e993c17725c51c03a4cd9630603298ad3c385765))

## [0.24.0](https://github.com/Safe-DS/Library/compare/v0.23.0...v0.24.0) (2024-05-09)


### Features

* `Column.plot_histogram()` using `Table.plot_histograms` for consistent results ([#726](https://github.com/Safe-DS/Library/issues/726)) ([576492c](https://github.com/Safe-DS/Library/commit/576492cde80de2be4ab16b9926106beab13872f1))
* `Regressor.summarize_metrics` and `Classifier.summarize_metrics` ([#729](https://github.com/Safe-DS/Library/issues/729)) ([1cc14b1](https://github.com/Safe-DS/Library/commit/1cc14b16a5394e98bd64ad0aa39562af5c89b94d)), closes [#713](https://github.com/Safe-DS/Library/issues/713)
* `Table.keep_only_rows` ([#721](https://github.com/Safe-DS/Library/issues/721)) ([923a6c2](https://github.com/Safe-DS/Library/commit/923a6c29f9c9b6dc59a10b5746d71c6cff029f69))
* `Table.remove_rows` ([#720](https://github.com/Safe-DS/Library/issues/720)) ([a1cdaef](https://github.com/Safe-DS/Library/commit/a1cdaef259e9f889f8fb0355ef69d4666faa9c94)), closes [#698](https://github.com/Safe-DS/Library/issues/698)
* Add `ImageDataset` and Layer for ConvolutionalNeuralNetworks ([#645](https://github.com/Safe-DS/Library/issues/645)) ([5b6d219](https://github.com/Safe-DS/Library/commit/5b6d21988dd4513add7ef0068670d7b0cb9d6a9f)), closes [#579](https://github.com/Safe-DS/Library/issues/579) [#580](https://github.com/Safe-DS/Library/issues/580) [#581](https://github.com/Safe-DS/Library/issues/581)
* added load_percentage parameter to ImageList.from_files to load a subset of the given files ([#739](https://github.com/Safe-DS/Library/issues/739)) ([0564b52](https://github.com/Safe-DS/Library/commit/0564b523c46df74ac235c22783dcbf3d515ed122)), closes [#736](https://github.com/Safe-DS/Library/issues/736)
* added rnn layer and TimeSeries conversion ([#615](https://github.com/Safe-DS/Library/issues/615)) ([6cad203](https://github.com/Safe-DS/Library/commit/6cad203c951bd1a6f11bbaa8c513df6ed0ad226e)), closes [#614](https://github.com/Safe-DS/Library/issues/614) [#648](https://github.com/Safe-DS/Library/issues/648) [#656](https://github.com/Safe-DS/Library/issues/656) [#601](https://github.com/Safe-DS/Library/issues/601)
* Basic implementation of cell with polars ([#734](https://github.com/Safe-DS/Library/issues/734)) ([004630b](https://github.com/Safe-DS/Library/commit/004630b8195f3fca0d04311186c71abe486b8bf2)), closes [#712](https://github.com/Safe-DS/Library/issues/712)
* deprecate `Table.add_column` and `Table.add_row` ([#723](https://github.com/Safe-DS/Library/issues/723)) ([5dd9d02](https://github.com/Safe-DS/Library/commit/5dd9d026e404e02bfe0905e8328d5b621f36c219)), closes [#722](https://github.com/Safe-DS/Library/issues/722)
* deprecated `Table.from_excel_file` and `Table.to_excel_file` ([#728](https://github.com/Safe-DS/Library/issues/728)) ([c89e0bf](https://github.com/Safe-DS/Library/commit/c89e0bf4b2ecdd1d9be2694331c399994e565a06)), closes [#727](https://github.com/Safe-DS/Library/issues/727)
* Larger histogram plot if table only has one column ([#716](https://github.com/Safe-DS/Library/issues/716)) ([31ffd12](https://github.com/Safe-DS/Library/commit/31ffd12fbf3526f2150b67109931d1a103eb462d))
* polars implementation of a column ([#738](https://github.com/Safe-DS/Library/issues/738)) ([732aa48](https://github.com/Safe-DS/Library/commit/732aa480372fb8ce7c8d522c3f4ccddb4383b73f)), closes [#712](https://github.com/Safe-DS/Library/issues/712)
* polars implementation of a row ([#733](https://github.com/Safe-DS/Library/issues/733)) ([ff627f6](https://github.com/Safe-DS/Library/commit/ff627f66e45e3cdac5c5c956346251222566664d)), closes [#712](https://github.com/Safe-DS/Library/issues/712)
* polars implementation of table ([#744](https://github.com/Safe-DS/Library/issues/744)) ([fc49895](https://github.com/Safe-DS/Library/commit/fc498953ad30dbff5547557d35c06f8c3267cd79)), closes [#638](https://github.com/Safe-DS/Library/issues/638) [#641](https://github.com/Safe-DS/Library/issues/641) [#649](https://github.com/Safe-DS/Library/issues/649) [#712](https://github.com/Safe-DS/Library/issues/712)
* regularization for decision trees and random forests ([#730](https://github.com/Safe-DS/Library/issues/730)) ([102de2d](https://github.com/Safe-DS/Library/commit/102de2d90a129d33c78f58597d590df1f0da3ad3)), closes [#700](https://github.com/Safe-DS/Library/issues/700)
* Remove device information in image class ([#735](https://github.com/Safe-DS/Library/issues/735)) ([d783caa](https://github.com/Safe-DS/Library/commit/d783caa11bb2dc0e466e781eb55df6ed8a8b51a3)), closes [#524](https://github.com/Safe-DS/Library/issues/524)
* return fitted transformer and transformed table from `fit_and_transform` ([#724](https://github.com/Safe-DS/Library/issues/724)) ([2960d35](https://github.com/Safe-DS/Library/commit/2960d3558851691bb6986cbb05dcfa514b6f3ab6)), closes [#613](https://github.com/Safe-DS/Library/issues/613)


### Bug Fixes

* make `Image.clone` internal ([#725](https://github.com/Safe-DS/Library/issues/725)) ([215a472](https://github.com/Safe-DS/Library/commit/215a4726cd0e3615b95329f317433e3c48bd208d)), closes [#626](https://github.com/Safe-DS/Library/issues/626)


### Performance Improvements

* improved performance of `TabularDataset.__eq__` by a factor of up to 2 ([#697](https://github.com/Safe-DS/Library/issues/697)) ([cd7f55b](https://github.com/Safe-DS/Library/commit/cd7f55ba4f882c2af9233dece53d8e8b6191a6fb))

## [0.23.0](https://github.com/Safe-DS/Library/compare/v0.22.1...v0.23.0) (2024-05-04)


### Features

* add `Column.to_table` ([#705](https://github.com/Safe-DS/Library/issues/705)) ([36e4a7a](https://github.com/Safe-DS/Library/commit/36e4a7abc03bb826a4f55bf02575e45fd5412bf8)), closes [#695](https://github.com/Safe-DS/Library/issues/695)
* added Column.summarize_statistics() ([#715](https://github.com/Safe-DS/Library/issues/715)) ([71730a9](https://github.com/Safe-DS/Library/commit/71730a99a8af16b3e1e0e3c478250768785ca4d9)), closes [#701](https://github.com/Safe-DS/Library/issues/701)
* replace other values than NaN with imputer ([#707](https://github.com/Safe-DS/Library/issues/707)) ([4a059e0](https://github.com/Safe-DS/Library/commit/4a059e09a4c66397cf266f71c4248543e2e972e5)), closes [#643](https://github.com/Safe-DS/Library/issues/643)


### Bug Fixes

* use UTF-8 encoding when opening files ([#704](https://github.com/Safe-DS/Library/issues/704)) ([f8c27bc](https://github.com/Safe-DS/Library/commit/f8c27bcd5e1edb0243a1d785740b1883b477db4b))

## [0.22.1](https://github.com/Safe-DS/Library/compare/v0.22.0...v0.22.1) (2024-05-02)


### Bug Fixes

* make `_Layer`, `_InputConversion`, `_OutputConversion` public ([#692](https://github.com/Safe-DS/Library/issues/692)) ([eb226f4](https://github.com/Safe-DS/Library/commit/eb226f4e83267ab6c3b1fa1b862aeda8a55b1ff8)), closes [#690](https://github.com/Safe-DS/Library/issues/690) [#691](https://github.com/Safe-DS/Library/issues/691)

## [0.22.0](https://github.com/Safe-DS/Library/compare/v0.21.0...v0.22.0) (2024-05-01)


### Features

* `is_fitted` is now always a property ([#662](https://github.com/Safe-DS/Library/issues/662)) ([b1db881](https://github.com/Safe-DS/Library/commit/b1db8812f3105e93f9b1423d51c785499014bbaa)), closes [#586](https://github.com/Safe-DS/Library/issues/586)
* add `Column.missing_value_count` ([#682](https://github.com/Safe-DS/Library/issues/682)) ([f084916](https://github.com/Safe-DS/Library/commit/f08491689552132a2be93bbe923cb15c3010ee54)), closes [#642](https://github.com/Safe-DS/Library/issues/642)
* Add `InputConversion` & `OutputConversion` for nn interface ([#625](https://github.com/Safe-DS/Library/issues/625)) ([fd723f7](https://github.com/Safe-DS/Library/commit/fd723f7fc9198e4b8c62b9c5ebefe4e5dac81ad3)), closes [#621](https://github.com/Safe-DS/Library/issues/621)
* Add hash,eq and sizeof in ForwardLayer ([#634](https://github.com/Safe-DS/Library/issues/634)) ([72f7fde](https://github.com/Safe-DS/Library/commit/72f7fdea123645d3c4f9d304fbd249bf5e5f8ab3)), closes [#633](https://github.com/Safe-DS/Library/issues/633)
* allow using tables that already contain target for prediction ([#687](https://github.com/Safe-DS/Library/issues/687)) ([e9f1cfb](https://github.com/Safe-DS/Library/commit/e9f1cfb72bfc952d7be149056dc1d6939e443d88)), closes [#636](https://github.com/Safe-DS/Library/issues/636)
* callback `Row.sort_columns` takes four parameters instead of two tuples ([#683](https://github.com/Safe-DS/Library/issues/683)) ([9c3e3de](https://github.com/Safe-DS/Library/commit/9c3e3de6b458c29cc76b14ac8131df008cb5a7e2)), closes [#584](https://github.com/Safe-DS/Library/issues/584)
* rename `group_rows_by` in `Table` to `group_rows` ([#661](https://github.com/Safe-DS/Library/issues/661)) ([c1644b7](https://github.com/Safe-DS/Library/commit/c1644b72763d302e91bf3e6064659ac5f00a2730)), closes [#611](https://github.com/Safe-DS/Library/issues/611)
* rename `number_of_column` in `Row` to `number_of_columns` ([#660](https://github.com/Safe-DS/Library/issues/660)) ([0a08296](https://github.com/Safe-DS/Library/commit/0a08296d25de673e9995f5aec7c808d2e25996b5)), closes [#646](https://github.com/Safe-DS/Library/issues/646)
* rework `TaggedTable` ([#680](https://github.com/Safe-DS/Library/issues/680)) ([db2b613](https://github.com/Safe-DS/Library/commit/db2b61365b4a035d9b58f7725f82836821e08b71)), closes [#647](https://github.com/Safe-DS/Library/issues/647)
* show missing value count/ratio in summarized statistics ([#684](https://github.com/Safe-DS/Library/issues/684)) ([74b8a35](https://github.com/Safe-DS/Library/commit/74b8a3528bfa8315eb617e269cdc6cfd718d0e8f)), closes [#619](https://github.com/Safe-DS/Library/issues/619)
* specify `extras` instead of `features` in `to_tabular_dataset` ([#685](https://github.com/Safe-DS/Library/issues/685)) ([841657f](https://github.com/Safe-DS/Library/commit/841657f6f7ba7b726fc11dda0c0678d91551adbb)), closes [#623](https://github.com/Safe-DS/Library/issues/623)


### Bug Fixes

* actually use `kernel` of support vector machines for training ([#681](https://github.com/Safe-DS/Library/issues/681)) ([09c5082](https://github.com/Safe-DS/Library/commit/09c5082215b399a4853fe6dac2a0e44a7915f8f5)), closes [#602](https://github.com/Safe-DS/Library/issues/602)


### Performance Improvements

* Faster plot_histograms and more reliable plots ([#659](https://github.com/Safe-DS/Library/issues/659)) ([b5f0a12](https://github.com/Safe-DS/Library/commit/b5f0a12c4d406a4dbe6f92d313be9ae537cb47f8))

## [0.21.0](https://github.com/Safe-DS/Library/compare/v0.20.0...v0.21.0) (2024-04-17)


### Features

* add ARIMA model ([#577](https://github.com/Safe-DS/Library/issues/577)) ([8b9c7a9](https://github.com/Safe-DS/Library/commit/8b9c7a9295a2bfc0c425d247e76a38b4b79136b8)), closes [#570](https://github.com/Safe-DS/Library/issues/570)
* Add ImageList class ([#534](https://github.com/Safe-DS/Library/issues/534)) ([3cb74a2](https://github.com/Safe-DS/Library/commit/3cb74a2a0acc0bdeb795c2d22d1cc02e9fb6b790)), closes [#528](https://github.com/Safe-DS/Library/issues/528) [#599](https://github.com/Safe-DS/Library/issues/599) [#600](https://github.com/Safe-DS/Library/issues/600)
* more hash, sizeof and eq implementations ([#609](https://github.com/Safe-DS/Library/issues/609)) ([2bc0b0a](https://github.com/Safe-DS/Library/commit/2bc0b0a2ba0a5eda0d4c220210c59247d3a18ef2))


### Performance Improvements

* Add special case to `Table.add_rows` to increase performance ([#608](https://github.com/Safe-DS/Library/issues/608)) ([ffb8304](https://github.com/Safe-DS/Library/commit/ffb83042307dbe77baef6e24dbefb9691b02db15)), closes [#606](https://github.com/Safe-DS/Library/issues/606)
* improve performance of model & forward layer ([#616](https://github.com/Safe-DS/Library/issues/616)) ([e856cd5](https://github.com/Safe-DS/Library/commit/e856cd50f806b84098a1b6caf206ccfb9955df58)), closes [#610](https://github.com/Safe-DS/Library/issues/610)
* lazily import our modules and external libraries ([#624](https://github.com/Safe-DS/Library/issues/624)) ([20fc313](https://github.com/Safe-DS/Library/commit/20fc31395390a21ce052bea6dc295ed66ddef310))
* treat Tables specially when calling add_rows ([#606](https://github.com/Safe-DS/Library/issues/606)) ([e555b85](https://github.com/Safe-DS/Library/commit/e555b85dbd3a5e967f6337faa3d2ac4a9e8d8208)), closes [#575](https://github.com/Safe-DS/Library/issues/575)

## [0.20.0](https://github.com/Safe-DS/Library/compare/v0.19.0...v0.20.0) (2024-04-03)


### Features

* add deterministic hash methods to all types ([#573](https://github.com/Safe-DS/Library/issues/573)) ([f6a3ca7](https://github.com/Safe-DS/Library/commit/f6a3ca791fc1b3b99266dbe68c85dde0669c4186))
* add fnn functionality ([#529](https://github.com/Safe-DS/Library/issues/529)) ([ce53153](https://github.com/Safe-DS/Library/commit/ce5315308b6d1c56258397c1300ee64698cf71bb)), closes [#522](https://github.com/Safe-DS/Library/issues/522)
* add suffixes to models to indicate their task ([#588](https://github.com/Safe-DS/Library/issues/588)) ([d490dee](https://github.com/Safe-DS/Library/commit/d490dee5e01d65216d4417cceef505c08b27fda3))
* added lag_plot ([#548](https://github.com/Safe-DS/Library/issues/548)) ([0fb38d2](https://github.com/Safe-DS/Library/commit/0fb38d252c1596063ddad768d8a0a1e6ad07c1d4)), closes [#519](https://github.com/Safe-DS/Library/issues/519)
* added normal plot for time series ([#550](https://github.com/Safe-DS/Library/issues/550)) ([dbdf11e](https://github.com/Safe-DS/Library/commit/dbdf11e9c8cbdd39861ba2c7946caf0e135011d2)), closes [#549](https://github.com/Safe-DS/Library/issues/549)
* when using from table to time series feature must be given ([#572](https://github.com/Safe-DS/Library/issues/572)) ([ca23f0f](https://github.com/Safe-DS/Library/commit/ca23f0f6f6f42a36f79a38d1a7a696ca523bfd0c)), closes [#571](https://github.com/Safe-DS/Library/issues/571)


### Bug Fixes

* incorrect type hint for `number_of_bins` parameter ([#567](https://github.com/Safe-DS/Library/issues/567)) ([b434e53](https://github.com/Safe-DS/Library/commit/b434e537661e56fec44e265ef05f3e78779effc5))
* mark various API elements as internal ([#587](https://github.com/Safe-DS/Library/issues/587)) ([ea176fc](https://github.com/Safe-DS/Library/commit/ea176fc576dd715dbfd2699055770a80d5ffb5ad)), closes [#582](https://github.com/Safe-DS/Library/issues/582) [#585](https://github.com/Safe-DS/Library/issues/585)

## [0.19.0](https://github.com/Safe-DS/Library/compare/v0.18.0...v0.19.0) (2024-02-06)


### Features

* return the correct size for custom container objects ([#547](https://github.com/Safe-DS/Library/issues/547)) ([f44c34d](https://github.com/Safe-DS/Library/commit/f44c34d2be12d4887f662320c332c58cb3ac4220))

## [0.18.0](https://github.com/Safe-DS/Library/compare/v0.17.1...v0.18.0) (2024-02-03)


### Features

* Add adjust_color_balance method in Image ([#530](https://github.com/Safe-DS/Library/issues/530)) ([dba23f9](https://github.com/Safe-DS/Library/commit/dba23f90b29af3ac5b3187eeea53dcedebe9a2b8)), closes [#525](https://github.com/Safe-DS/Library/issues/525)
* Add find_edges method in Image ([#531](https://github.com/Safe-DS/Library/issues/531)) ([d728eb6](https://github.com/Safe-DS/Library/commit/d728eb60aa75afddde386a129f93cd0f327e56df)), closes [#523](https://github.com/Safe-DS/Library/issues/523)
* class for time series ([#508](https://github.com/Safe-DS/Library/issues/508)) ([73cdfb1](https://github.com/Safe-DS/Library/commit/73cdfb119cb4243e56a5e99d40cfdbacf0466daf)), closes [#481](https://github.com/Safe-DS/Library/issues/481)

## [0.17.1](https://github.com/Safe-DS/Library/compare/v0.17.0...v0.17.1) (2024-01-11)


### Bug Fixes

* update `torch` ([#527](https://github.com/Safe-DS/Library/issues/527)) ([6934d1d](https://github.com/Safe-DS/Library/commit/6934d1d1820e9e0d1b8877cafdffc449595dca29))

## [0.17.0](https://github.com/Safe-DS/Library/compare/v0.16.0...v0.17.0) (2024-01-11)


### Features

* change image class to use PyTorch tensors and methods ([#506](https://github.com/Safe-DS/Library/issues/506)) ([efa2b61](https://github.com/Safe-DS/Library/commit/efa2b61c4c1d3a54f8b94ddb2cb62792a03c1b85)), closes [#505](https://github.com/Safe-DS/Library/issues/505)

## [0.16.0](https://github.com/Safe-DS/Library/compare/v0.15.0...v0.16.0) (2023-11-22)


### Features

* drop Python 3.10 and add Python 3.12 ([#478](https://github.com/Safe-DS/Library/issues/478)) ([5bf0e75](https://github.com/Safe-DS/Library/commit/5bf0e75746662f91cc354d1a63882ad3abf5b84e))
* enable copy-on-write for pandas dataframes ([#494](https://github.com/Safe-DS/Library/issues/494)) ([9a19389](https://github.com/Safe-DS/Library/commit/9a1938989c084414df5ee6a8510957760cfe321e)), closes [#428](https://github.com/Safe-DS/Library/issues/428)


### Bug Fixes

* **deps-dev:** Bump urllib3 from 1.26.17 to 1.26.18 ([#480](https://github.com/Safe-DS/Library/issues/480)) ([a592d2c](https://github.com/Safe-DS/Library/commit/a592d2c2e7d93127a80159945f436fcb86b81620)), closes [#3159](https://github.com/Safe-DS/Library/issues/3159)


### Performance Improvements

* remove unneeded copy operations in table transformers ([#496](https://github.com/Safe-DS/Library/issues/496)) ([6443beb](https://github.com/Safe-DS/Library/commit/6443beb28e0dc8cf6f9f2508872dd67a6abde363)), closes [#494](https://github.com/Safe-DS/Library/issues/494)

## [0.15.0](https://github.com/Safe-DS/Library/compare/v0.14.0...v0.15.0) (2023-07-13)


### Features

* Add copy method for tables ([#405](https://github.com/Safe-DS/Library/issues/405)) ([72e87f0](https://github.com/Safe-DS/Library/commit/72e87f0e3fd8c647b1021678ed8712224de074de)), closes [#275](https://github.com/Safe-DS/Library/issues/275)
* add gaussian noise to image ([#430](https://github.com/Safe-DS/Library/issues/430)) ([925a505](https://github.com/Safe-DS/Library/commit/925a50547cb8726ac3038fa7c4108eb49f309cca)), closes [#381](https://github.com/Safe-DS/Library/issues/381)
* add schema conversions when adding new rows to a table and schema conversion when creating a new table ([#432](https://github.com/Safe-DS/Library/issues/432)) ([6e9ff69](https://github.com/Safe-DS/Library/commit/6e9ff6900b912a6c8f1dca8cafbb2f290fd146a5)), closes [#404](https://github.com/Safe-DS/Library/issues/404) [#322](https://github.com/Safe-DS/Library/issues/322) [#127](https://github.com/Safe-DS/Library/issues/127) [#322](https://github.com/Safe-DS/Library/issues/322) [#127](https://github.com/Safe-DS/Library/issues/127)
* add test for empty tables for the method `Table.sort_rows` ([#431](https://github.com/Safe-DS/Library/issues/431)) ([f94b768](https://github.com/Safe-DS/Library/commit/f94b768ae089988ec9511e2356a9f201ba41fea8)), closes [#402](https://github.com/Safe-DS/Library/issues/402)
* added color adjustment feature ([#409](https://github.com/Safe-DS/Library/issues/409)) ([2cbee36](https://github.com/Safe-DS/Library/commit/2cbee36d3bfcfb8389731ba4d20687587a781a7a)), closes [#380](https://github.com/Safe-DS/Library/issues/380)
* added test_repr table tests ([#410](https://github.com/Safe-DS/Library/issues/410)) ([cb77790](https://github.com/Safe-DS/Library/commit/cb777906e81f2becf3da2b530dd2cb84ad42fd63)), closes [#349](https://github.com/Safe-DS/Library/issues/349)
* discretize table ([#327](https://github.com/Safe-DS/Library/issues/327)) ([5e3da8d](https://github.com/Safe-DS/Library/commit/5e3da8d26250b9f06a3209a35e26e296890fc6a2)), closes [#143](https://github.com/Safe-DS/Library/issues/143)
* Improve error handling of TaggedTable ([#450](https://github.com/Safe-DS/Library/issues/450)) ([c5da544](https://github.com/Safe-DS/Library/commit/c5da544554d1895aad274ba5a4775fd63257d65f)), closes [#150](https://github.com/Safe-DS/Library/issues/150)
* Maintain tagging in methods inherited from `Table` class ([#332](https://github.com/Safe-DS/Library/issues/332)) ([bc73a6c](https://github.com/Safe-DS/Library/commit/bc73a6ca1c4a8429ced3a3b27ff5dedbeec59c03)), closes [#58](https://github.com/Safe-DS/Library/issues/58)
* new error class `OutOfBoundsError` ([#438](https://github.com/Safe-DS/Library/issues/438)) ([1f37e4a](https://github.com/Safe-DS/Library/commit/1f37e4a18b8637ebcb6a5db22642d13c5568cd1a)), closes [#262](https://github.com/Safe-DS/Library/issues/262)
* rename several `Table` methods for consistency ([#445](https://github.com/Safe-DS/Library/issues/445)) ([9954986](https://github.com/Safe-DS/Library/commit/9954986b1a5a3bf11c70cf9a82538b48ff840e12)), closes [#439](https://github.com/Safe-DS/Library/issues/439)
* suggest similar columns if column gets accessed that doesnt exist ([#385](https://github.com/Safe-DS/Library/issues/385)) ([6a097a4](https://github.com/Safe-DS/Library/commit/6a097a4d3e6544a0c6c96b44d5c461ac2cbc61e8)), closes [#203](https://github.com/Safe-DS/Library/issues/203)


### Bug Fixes

* added the missing ids in parameterized tests ([#412](https://github.com/Safe-DS/Library/issues/412)) ([dab6419](https://github.com/Safe-DS/Library/commit/dab64191791550f9449e2fd4b44b8d4228e70f34)), closes [#362](https://github.com/Safe-DS/Library/issues/362)
* don't warn if `Imputer` transforms column without missing values ([#448](https://github.com/Safe-DS/Library/issues/448)) ([f0cb6a5](https://github.com/Safe-DS/Library/commit/f0cb6a5d0852f280b1b88f62e16c2014b09f0c65))
* Warnings raised by underlying seaborn and numpy libraries  ([#425](https://github.com/Safe-DS/Library/issues/425)) ([c4143af](https://github.com/Safe-DS/Library/commit/c4143afed0f3345baae4052427eed2e3e0d296f4)), closes [#357](https://github.com/Safe-DS/Library/issues/357)

## [0.14.0](https://github.com/Safe-DS/Library/compare/v0.13.0...v0.14.0) (2023-06-30)


### Features

* Add `find_edges` method to `Image` class ([#383](https://github.com/Safe-DS/Library/issues/383)) ([d14b6ce](https://github.com/Safe-DS/Library/commit/d14b6ce281c8b1324f727c45ca13b99b6f9ed2a2)), closes [#288](https://github.com/Safe-DS/Library/issues/288)
* Add `StandardScaler` transformer ([#316](https://github.com/Safe-DS/Library/issues/316)) ([57b0572](https://github.com/Safe-DS/Library/commit/57b057289ee5c862422df7d85e4fe72047567b7f)), closes [#142](https://github.com/Safe-DS/Library/issues/142)
* Add docstrings to the getter methods for hyperparameters in Regression and Classification models ([#371](https://github.com/Safe-DS/Library/issues/371)) ([9073f04](https://github.com/Safe-DS/Library/commit/9073f04cd339fc2146ea04339ed14688f29b336f)), closes [#313](https://github.com/Safe-DS/Library/issues/313)
* Added `Table.group_by` to group a table by a given key ([#343](https://github.com/Safe-DS/Library/issues/343)) ([afb98be](https://github.com/Safe-DS/Library/commit/afb98be2b1d0db0dad2084979537a1e5d449ba07)), closes [#160](https://github.com/Safe-DS/Library/issues/160)
* Added and improved errors and warnings in the table transformers ([#372](https://github.com/Safe-DS/Library/issues/372)) ([544e307](https://github.com/Safe-DS/Library/commit/544e307af638f8d830171ac34e3edb61da9abc07)), closes [#152](https://github.com/Safe-DS/Library/issues/152)
* added crop() method in image and tests ([#365](https://github.com/Safe-DS/Library/issues/365)) ([eba8163](https://github.com/Safe-DS/Library/commit/eba8163d649d1e4dcdda2ff4fc3e2229249d8940))
* added invert_colors method ([#367](https://github.com/Safe-DS/Library/issues/367)) ([1e4d110](https://github.com/Safe-DS/Library/commit/1e4d110da9516fd1f95f2e9d7737ebeef250eb6a))
* adjust brightness and contrast of image ([#368](https://github.com/Safe-DS/Library/issues/368)) ([1752feb](https://github.com/Safe-DS/Library/commit/1752feb06bb5b0f041310956eca2ee5ed08dd999)), closes [#289](https://github.com/Safe-DS/Library/issues/289) [#291](https://github.com/Safe-DS/Library/issues/291)
* blur Image method ([#363](https://github.com/Safe-DS/Library/issues/363)) ([c642176](https://github.com/Safe-DS/Library/commit/c6421762c7ec810137713b581f71e0b1a66176c0))
* check that methods of table can handle an empty table ([#314](https://github.com/Safe-DS/Library/issues/314)) ([686c2e7](https://github.com/Safe-DS/Library/commit/686c2e7728850d9c775135b391551495f6e87451)), closes [#123](https://github.com/Safe-DS/Library/issues/123)
* convert image to grayscale ([#366](https://github.com/Safe-DS/Library/issues/366)) ([1312fe7](https://github.com/Safe-DS/Library/commit/1312fe76dd3e79506423058b8b771788be8b54a3)), closes [#287](https://github.com/Safe-DS/Library/issues/287)
* enhance `replace_column` to accept a list of new columns ([#312](https://github.com/Safe-DS/Library/issues/312)) ([d50c5b5](https://github.com/Safe-DS/Library/commit/d50c5b54f3c1ec70eda2fd9bfcc7148fa3e89c6b)), closes [#301](https://github.com/Safe-DS/Library/issues/301)
* Explicitly throw `UnknownColumnNameError` in `TaggedTable._from_table` ([#334](https://github.com/Safe-DS/Library/issues/334)) ([498999f](https://github.com/Safe-DS/Library/commit/498999f50d9f48609043f87f45ec9383ace3afe1)), closes [#333](https://github.com/Safe-DS/Library/issues/333)
* flip images / eq method for image ([#360](https://github.com/Safe-DS/Library/issues/360)) ([54f4ae1](https://github.com/Safe-DS/Library/commit/54f4ae183a0d5e6f18386cc255594f8c810c0fdf)), closes [#280](https://github.com/Safe-DS/Library/issues/280)
* improve `table.summary`. Catch `ValueError` thrown by `column.stability` ([#390](https://github.com/Safe-DS/Library/issues/390)) ([dbbe0e3](https://github.com/Safe-DS/Library/commit/dbbe0e3a6ebd2f22c386c13a3cc53184e531f837)), closes [#320](https://github.com/Safe-DS/Library/issues/320)
* improve error handling of `column.stability` when given a column that contains only None ([#388](https://github.com/Safe-DS/Library/issues/388)) ([1da2499](https://github.com/Safe-DS/Library/commit/1da24992d02a3cbcea12fef0b78004d99403beb0)), closes [#319](https://github.com/Safe-DS/Library/issues/319)
* Improve Error Handling of classifiers and regressors ([#355](https://github.com/Safe-DS/Library/issues/355)) ([66f5f64](https://github.com/Safe-DS/Library/commit/66f5f647820b3bf36a6041f04d6e8547170b1c81)), closes [#153](https://github.com/Safe-DS/Library/issues/153)
* properties for width-height of image ([#359](https://github.com/Safe-DS/Library/issues/359)) ([d9ebdc1](https://github.com/Safe-DS/Library/commit/d9ebdc1e3deef986026abf5d90c57376dbac23ee)), closes [#290](https://github.com/Safe-DS/Library/issues/290)
* Resize image ([#354](https://github.com/Safe-DS/Library/issues/354)) ([3a971ca](https://github.com/Safe-DS/Library/commit/3a971ca3c88b86d7a780d3f163a2ee86ab35b98c)), closes [#283](https://github.com/Safe-DS/Library/issues/283)
* rotate_left and rotate_right added to Image ([#361](https://github.com/Safe-DS/Library/issues/361)) ([c877530](https://github.com/Safe-DS/Library/commit/c8775306deffe826cb14d0fe177e0c09aa4204fd)), closes [#281](https://github.com/Safe-DS/Library/issues/281)
* set kernel of support vector machine ([#350](https://github.com/Safe-DS/Library/issues/350)) ([1326f40](https://github.com/Safe-DS/Library/commit/1326f40a260437a373711c7c65a986bcfa321980)), closes [#172](https://github.com/Safe-DS/Library/issues/172)
* sharpen image ([#364](https://github.com/Safe-DS/Library/issues/364)) ([3444700](https://github.com/Safe-DS/Library/commit/344470004199dcf6763b1145ce38fdf14e11483a)), closes [#286](https://github.com/Safe-DS/Library/issues/286)


### Bug Fixes

* Keeping no columns with Table.keep_only_columns results in an empty Table with a row count above 0 ([#386](https://github.com/Safe-DS/Library/issues/386)) ([15dab06](https://github.com/Safe-DS/Library/commit/15dab06d87a8f8cff7e676b67145eff55824134a)), closes [#318](https://github.com/Safe-DS/Library/issues/318)
* remove default value of `positive_class` parameter of classifier metrics ([#382](https://github.com/Safe-DS/Library/issues/382)) ([58fc09e](https://github.com/Safe-DS/Library/commit/58fc09eab2db1fb678522a807a8a20ed519627fd))
* remove default value of `radius` parameter of `blur` ([#378](https://github.com/Safe-DS/Library/issues/378)) ([7f07f29](https://github.com/Safe-DS/Library/commit/7f07f29ad0b426861b929297305e61e0c2d93ebc))

## [0.13.0](https://github.com/Safe-DS/Library/compare/v0.12.0...v0.13.0) (2023-06-01)


### Features

* add `Choice` class for possible values of hyperparameter ([#325](https://github.com/Safe-DS/Library/issues/325)) ([d511c3e](https://github.com/Safe-DS/Library/commit/d511c3eed779e64fc53499e7c2eb2e8292955645)), closes [#264](https://github.com/Safe-DS/Library/issues/264)
* Add `RangeScaler` transformer ([#310](https://github.com/Safe-DS/Library/issues/310)) ([f687840](https://github.com/Safe-DS/Library/commit/f68784057afe20d5450e9eb875fce1a07fb5fa77)), closes [#141](https://github.com/Safe-DS/Library/issues/141)
* Add methods that tell which columns would be affected by a transformer ([#304](https://github.com/Safe-DS/Library/issues/304)) ([3933b45](https://github.com/Safe-DS/Library/commit/3933b458042f524d337f41d0ffa3aa4da16f5a2e)), closes [#190](https://github.com/Safe-DS/Library/issues/190)
* Getters for hyperparameters of Regression and Classification models ([#306](https://github.com/Safe-DS/Library/issues/306)) ([5c7a662](https://github.com/Safe-DS/Library/commit/5c7a6623cd47f7c6cc25d2cd02179ff5b1a520d9)), closes [#260](https://github.com/Safe-DS/Library/issues/260)
* improve error handling of table ([#308](https://github.com/Safe-DS/Library/issues/308)) ([ef87cc4](https://github.com/Safe-DS/Library/commit/ef87cc4d7f62fd0830688f8535e53ad7e2329457)), closes [#147](https://github.com/Safe-DS/Library/issues/147)
* Remove warnings thrown in new `Transformer` methods ([#324](https://github.com/Safe-DS/Library/issues/324)) ([ca046c4](https://github.com/Safe-DS/Library/commit/ca046c40217bebcc05af98129d4e194c0509c9bb)), closes [#323](https://github.com/Safe-DS/Library/issues/323)

## [0.12.0](https://github.com/Safe-DS/Library/compare/v0.11.0...v0.12.0) (2023-05-11)


### Features

* add `learning_rate` to AdaBoost classifier and regressor. ([#251](https://github.com/Safe-DS/Library/issues/251)) ([7f74440](https://github.com/Safe-DS/Library/commit/7f744409c08fb465d59f1f04e2cac7ebed23f339)), closes [#167](https://github.com/Safe-DS/Library/issues/167)
* add alpha parameter to `lasso_regression` ([#232](https://github.com/Safe-DS/Library/issues/232)) ([b5050b9](https://github.com/Safe-DS/Library/commit/b5050b91f17774fa5cf3fc80b51d3ea6c295481f)), closes [#163](https://github.com/Safe-DS/Library/issues/163)
* add parameter `lasso_ratio` to `ElasticNetRegression` ([#237](https://github.com/Safe-DS/Library/issues/237)) ([4a1a736](https://github.com/Safe-DS/Library/commit/4a1a7367099125d2a072bf36686063de7180e8f0)), closes [#166](https://github.com/Safe-DS/Library/issues/166)
* Add parameter `number_of_tree` to `RandomForest` classifier and regressor ([#230](https://github.com/Safe-DS/Library/issues/230)) ([414336a](https://github.com/Safe-DS/Library/commit/414336ac9752f961cab30545cbe51befbde50d21)), closes [#161](https://github.com/Safe-DS/Library/issues/161)
* Added `Table.plot_boxplots` to plot a boxplot for each numerical column in the table ([#254](https://github.com/Safe-DS/Library/issues/254)) ([0203a0c](https://github.com/Safe-DS/Library/commit/0203a0c977184cdee1769d317fcb1f7cb5c644f3)), closes [#156](https://github.com/Safe-DS/Library/issues/156) [#239](https://github.com/Safe-DS/Library/issues/239)
* Added `Table.plot_histograms` to plot a histogram for each column in the table ([#252](https://github.com/Safe-DS/Library/issues/252)) ([e27d410](https://github.com/Safe-DS/Library/commit/e27d410085ebaf9ab98069a5b175d800259d95a3)), closes [#157](https://github.com/Safe-DS/Library/issues/157)
* Added `Table.transform_table` method which returns the transformed Table ([#229](https://github.com/Safe-DS/Library/issues/229)) ([0a9ce72](https://github.com/Safe-DS/Library/commit/0a9ce72ba2101f99fea43dcd43b1f498dbb8e558)), closes [#110](https://github.com/Safe-DS/Library/issues/110)
* Added alpha parameter to `RidgeRegression` ([#231](https://github.com/Safe-DS/Library/issues/231)) ([1ddc948](https://github.com/Safe-DS/Library/commit/1ddc948aac5f153f649c3869b99184c2c1d96d9f)), closes [#164](https://github.com/Safe-DS/Library/issues/164)
* Added Column#transform ([#270](https://github.com/Safe-DS/Library/issues/270)) ([40fb756](https://github.com/Safe-DS/Library/commit/40fb7566307b4c015f3acae7bb94f4e937977e07)), closes [#255](https://github.com/Safe-DS/Library/issues/255)
* Added method `Table.inverse_transform_table` which returns the original table ([#227](https://github.com/Safe-DS/Library/issues/227)) ([846bf23](https://github.com/Safe-DS/Library/commit/846bf233235b2cdaf9bbd00cacb89ea44e94011b)), closes [#111](https://github.com/Safe-DS/Library/issues/111)
* Added parameter `c` to `SupportVectorMachines` ([#267](https://github.com/Safe-DS/Library/issues/267)) ([a88eb8b](https://github.com/Safe-DS/Library/commit/a88eb8b8c3f67e8485ce2847c4923a2cf0506f68)), closes [#169](https://github.com/Safe-DS/Library/issues/169)
* Added parameter `maximum_number_of_learner` and `learner` to `AdaBoost` ([#269](https://github.com/Safe-DS/Library/issues/269)) ([bb5a07e](https://github.com/Safe-DS/Library/commit/bb5a07e17b6563d394d79b62349633791675346f)), closes [#171](https://github.com/Safe-DS/Library/issues/171) [#173](https://github.com/Safe-DS/Library/issues/173)
* Added parameter `number_of_trees` to `GradientBoosting` ([#268](https://github.com/Safe-DS/Library/issues/268)) ([766f2ff](https://github.com/Safe-DS/Library/commit/766f2ff2a6d68098be3e858ad12bf9e509e5f192)), closes [#170](https://github.com/Safe-DS/Library/issues/170)
* Allow arguments of type pathlib.Path for file I/O methods ([#228](https://github.com/Safe-DS/Library/issues/228)) ([2b58c82](https://github.com/Safe-DS/Library/commit/2b58c82f50ce88b4778f3c82108f5d5f474fdfa9)), closes [#146](https://github.com/Safe-DS/Library/issues/146)
* convert `Schema` to `dict` and format it nicely in a notebook ([#244](https://github.com/Safe-DS/Library/issues/244)) ([ad1cac5](https://github.com/Safe-DS/Library/commit/ad1cac5198709d0a78019787251ba2aed913cf55)), closes [#151](https://github.com/Safe-DS/Library/issues/151)
* Convert between Excel file and `Table` ([#233](https://github.com/Safe-DS/Library/issues/233)) ([0d7a998](https://github.com/Safe-DS/Library/commit/0d7a998f9e660f47147f7eaa6ebb8119c09188ac)), closes [#138](https://github.com/Safe-DS/Library/issues/138) [#139](https://github.com/Safe-DS/Library/issues/139)
* convert containers for tabular data to HTML ([#243](https://github.com/Safe-DS/Library/issues/243)) ([683c279](https://github.com/Safe-DS/Library/commit/683c2793f053f5d0572e723b35db383aa00ddc44)), closes [#140](https://github.com/Safe-DS/Library/issues/140)
* make `Column` a subclass of `Sequence` ([#245](https://github.com/Safe-DS/Library/issues/245)) ([a35b943](https://github.com/Safe-DS/Library/commit/a35b943a126b28500499f5d7da1bccee10d98ff3))
* mark optional hyperparameters as keyword only ([#296](https://github.com/Safe-DS/Library/issues/296)) ([44a41eb](https://github.com/Safe-DS/Library/commit/44a41eb205ad0f69f01564ab318e53873bb902c4)), closes [#278](https://github.com/Safe-DS/Library/issues/278)
* move exceptions back to common package ([#295](https://github.com/Safe-DS/Library/issues/295)) ([a91172c](https://github.com/Safe-DS/Library/commit/a91172c0f21ea9934cedbe9fd749eb4ff7929394)), closes [#177](https://github.com/Safe-DS/Library/issues/177) [#262](https://github.com/Safe-DS/Library/issues/262)
* precision metric for classification ([#272](https://github.com/Safe-DS/Library/issues/272)) ([5adadad](https://github.com/Safe-DS/Library/commit/5adadadf6ab185b4d8864b7859d7cc036a055a6d)), closes [#185](https://github.com/Safe-DS/Library/issues/185)
* Raise error if an untagged table is used instead of a `TaggedTable` ([#234](https://github.com/Safe-DS/Library/issues/234)) ([8eea3dd](https://github.com/Safe-DS/Library/commit/8eea3dd31dab49b4d9371f61f02ace9fdca25394)), closes [#192](https://github.com/Safe-DS/Library/issues/192)
* recall and F1-score metrics for classification ([#277](https://github.com/Safe-DS/Library/issues/277)) ([2cf93cc](https://github.com/Safe-DS/Library/commit/2cf93cc7181ad69991055dd0e49035a785105356)), closes [#187](https://github.com/Safe-DS/Library/issues/187) [#186](https://github.com/Safe-DS/Library/issues/186)
* replace prefix `n` with `number_of` ([#250](https://github.com/Safe-DS/Library/issues/250)) ([f4f44a6](https://github.com/Safe-DS/Library/commit/f4f44a6b8d5f8ee795673b11c5f00e3ebb1b1b39)), closes [#171](https://github.com/Safe-DS/Library/issues/171)
* set `alpha` parameter for regularization of `ElasticNetRegression` ([#238](https://github.com/Safe-DS/Library/issues/238)) ([e642d1d](https://github.com/Safe-DS/Library/commit/e642d1d49c5b21240fa5bbbde48e80d5b7743ff1)), closes [#165](https://github.com/Safe-DS/Library/issues/165)
* Set `column_names` in `fit` methods of table transformers to be required ([#225](https://github.com/Safe-DS/Library/issues/225)) ([2856296](https://github.com/Safe-DS/Library/commit/2856296fb7228e8d4adebceb86e22ecaeb609ad9)), closes [#179](https://github.com/Safe-DS/Library/issues/179)
* set learning rate of Gradient Boosting models ([#253](https://github.com/Safe-DS/Library/issues/253)) ([9ffaf55](https://github.com/Safe-DS/Library/commit/9ffaf55a97333bb2edce2f2c9c66650a9724ca60)), closes [#168](https://github.com/Safe-DS/Library/issues/168)
* Support vector machine for regression and for classification ([#236](https://github.com/Safe-DS/Library/issues/236)) ([7f6c3bd](https://github.com/Safe-DS/Library/commit/7f6c3bd9fba670a487d3ef96d281f3904a8974a7)), closes [#154](https://github.com/Safe-DS/Library/issues/154)
* usable constructor for `Table` ([#294](https://github.com/Safe-DS/Library/issues/294)) ([56a1fc4](https://github.com/Safe-DS/Library/commit/56a1fc4450ba77877b6b29467c0e1d11dd200f9d)), closes [#266](https://github.com/Safe-DS/Library/issues/266)
* usable constructor for `TaggedTable` ([#299](https://github.com/Safe-DS/Library/issues/299)) ([01c3ad9](https://github.com/Safe-DS/Library/commit/01c3ad9564a35f31744a30862ae1a533ef5adf6b)), closes [#293](https://github.com/Safe-DS/Library/issues/293)


### Bug Fixes

* OneHotEncoder no longer creates duplicate column names ([#271](https://github.com/Safe-DS/Library/issues/271)) ([f604666](https://github.com/Safe-DS/Library/commit/f604666305d38d3a01696ea7ca60056ce7f78245)), closes [#201](https://github.com/Safe-DS/Library/issues/201)
* selectively ignore one warning instead of all warnings ([#235](https://github.com/Safe-DS/Library/issues/235)) ([3aad07d](https://github.com/Safe-DS/Library/commit/3aad07ddcc0da42e1dab2eed49fc41433a876765))

## [0.11.0](https://github.com/Safe-DS/Library/compare/v0.10.0...v0.11.0) (2023-04-21)


### Features

* `OneHotEncoder.inverse_transform` now maintains the column order from the original table ([#195](https://github.com/Safe-DS/Library/issues/195)) ([3ec0041](https://github.com/Safe-DS/Library/commit/3ec0041669ffe97640f96db345f3f43720d5c3f7)), closes [#109](https://github.com/Safe-DS/Library/issues/109)
* add `plot_` prefix back to plotting methods ([#212](https://github.com/Safe-DS/Library/issues/212)) ([e50c3b0](https://github.com/Safe-DS/Library/commit/e50c3b0118825e33eef0e2a7073673603e316ee5)), closes [#211](https://github.com/Safe-DS/Library/issues/211)
* adjust `Column`, `Schema` and `Table` to changes in `Row` ([#216](https://github.com/Safe-DS/Library/issues/216)) ([ca3eebb](https://github.com/Safe-DS/Library/commit/ca3eebbe2166f08d76cdcb89a012518ae8ff8e4e))
* back `Row` by a `polars.DataFrame` ([#214](https://github.com/Safe-DS/Library/issues/214)) ([62ca34d](https://github.com/Safe-DS/Library/commit/62ca34dd399da8b4e34b89bad408311707b53f41)), closes [#196](https://github.com/Safe-DS/Library/issues/196) [#149](https://github.com/Safe-DS/Library/issues/149)
* clean up `Row` class ([#215](https://github.com/Safe-DS/Library/issues/215)) ([b12fc68](https://github.com/Safe-DS/Library/commit/b12fc68c9b914446c1ea3aca5dacfab969680ae8))
* convert between `Row` and `dict` ([#206](https://github.com/Safe-DS/Library/issues/206)) ([e98b653](https://github.com/Safe-DS/Library/commit/e98b6536a2c50e64772fc7aeb29c03c850ebd570)), closes [#204](https://github.com/Safe-DS/Library/issues/204)
* convert between a `dict` and a `Table` ([#198](https://github.com/Safe-DS/Library/issues/198)) ([2a5089e](https://github.com/Safe-DS/Library/commit/2a5089e408a1eeb078aa77ce7c3e5ae8c4bc0201)), closes [#197](https://github.com/Safe-DS/Library/issues/197)
* create column types for `polars` data types ([#208](https://github.com/Safe-DS/Library/issues/208)) ([e18b362](https://github.com/Safe-DS/Library/commit/e18b36250ac170e3364106ba1c59649e0b4aff21)), closes [#196](https://github.com/Safe-DS/Library/issues/196)
* dataframe interchange protocol ([#200](https://github.com/Safe-DS/Library/issues/200)) ([bea976a](https://github.com/Safe-DS/Library/commit/bea976a72a28698a29145a3ad501d8af59e7e5d8)), closes [#199](https://github.com/Safe-DS/Library/issues/199)
* move existing ML solutions into `safeds.ml.classical` package ([#213](https://github.com/Safe-DS/Library/issues/213)) ([655f07f](https://github.com/Safe-DS/Library/commit/655f07f7f8f02b8fc92b469daf15a2384a81534f)), closes [#210](https://github.com/Safe-DS/Library/issues/210)


### Bug Fixes

* `table.keep_only_columns` now maps column names to correct data ([#194](https://github.com/Safe-DS/Library/issues/194)) ([459ab75](https://github.com/Safe-DS/Library/commit/459ab7570c7c7b79304f78cab4f6bff82fc026a3)), closes [#115](https://github.com/Safe-DS/Library/issues/115)
* typo in type hint ([#184](https://github.com/Safe-DS/Library/issues/184)) ([e79727d](https://github.com/Safe-DS/Library/commit/e79727d5d91090bc5cd6d3a81acc2a1393b3e5eb)), closes [#180](https://github.com/Safe-DS/Library/issues/180)

## [0.10.0](https://github.com/Safe-DS/Library/compare/v0.9.0...v0.10.0) (2023-04-13)


### Features

* move exceptions into subpackages ([#177](https://github.com/Safe-DS/Library/issues/177)) ([10b17fd](https://github.com/Safe-DS/Library/commit/10b17fddca6518dd0d62da0a791c508659c994c4))

## [0.9.0](https://github.com/Safe-DS/Library/compare/v0.8.0...v0.9.0) (2023-04-04)


### Features

* container for images ([#159](https://github.com/Safe-DS/Library/issues/159)) ([ed7ae34](https://github.com/Safe-DS/Library/commit/ed7ae341c4546ec32efe46e22dccc4d770126695)), closes [#158](https://github.com/Safe-DS/Library/issues/158)
* improve error handling for `predict` ([#145](https://github.com/Safe-DS/Library/issues/145)) ([a5ff11c](https://github.com/Safe-DS/Library/commit/a5ff11c2795e8e814b6a6565a9a331e1662d39c6)), closes [#9](https://github.com/Safe-DS/Library/issues/9)
* move `ImputerStrategy` to `safeds.data.tabular.typing` ([#174](https://github.com/Safe-DS/Library/issues/174)) ([205c8e2](https://github.com/Safe-DS/Library/commit/205c8e20ddcc76da57b895a23c0221da4dcf2508))
* rename `n_neighbors` to `number_of_neighbors` ([#162](https://github.com/Safe-DS/Library/issues/162)) ([526b96e](https://github.com/Safe-DS/Library/commit/526b96e3877299eb6bf6adea2882065fd29b52cf))


### Bug Fixes

* export `TableTransformer` and `InvertibleTableTransformer` ([#135](https://github.com/Safe-DS/Library/issues/135)) ([81c3695](https://github.com/Safe-DS/Library/commit/81c369556e8ca3bf800f843598efab29b0ac957b))

## [0.8.0](https://github.com/Safe-DS/Library/compare/v0.7.0...v0.8.0) (2023-03-31)


### Features

* create empty `Table` without schema ([#128](https://github.com/Safe-DS/Library/issues/128)) ([ddd3f59](https://github.com/Safe-DS/Library/commit/ddd3f59cf4f0173327511593ea4dc0b5b938fce1)), closes [#127](https://github.com/Safe-DS/Library/issues/127)
* improve `ColumnType`s ([#132](https://github.com/Safe-DS/Library/issues/132)) ([1786a87](https://github.com/Safe-DS/Library/commit/1786a872f9fe713b952e75c190245200082ac80d)), closes [#113](https://github.com/Safe-DS/Library/issues/113)
* infer schema of row if not passed explicitly ([#134](https://github.com/Safe-DS/Library/issues/134)) ([c5869bb](https://github.com/Safe-DS/Library/commit/c5869bbc215d884c48726b3c8f6b3556763d986d)), closes [#15](https://github.com/Safe-DS/Library/issues/15)
* new method `is_fitted` to check whether a model is fitted ([#130](https://github.com/Safe-DS/Library/issues/130)) ([8e1c3ea](https://github.com/Safe-DS/Library/commit/8e1c3ea22c3b422b65340f6fc25a87d0d7fb8869))
* new method `is_fitted` to check whether a transformer is fitted ([#131](https://github.com/Safe-DS/Library/issues/131)) ([e20954f](https://github.com/Safe-DS/Library/commit/e20954feb0f9191596aac93672b67361e1aa4ef8))
* rename `drop_XY` methods of `Table` to `remove_XY` ([#122](https://github.com/Safe-DS/Library/issues/122)) ([98d76a4](https://github.com/Safe-DS/Library/commit/98d76a46a56d4f78cb30d3ea8c4916b69f738674))
* rename `fit_transform` to `fit_and_transform` ([#119](https://github.com/Safe-DS/Library/issues/119)) ([76a7112](https://github.com/Safe-DS/Library/commit/76a71126b6ca21f9082dd2d3a2248bf65716b73f)), closes [#112](https://github.com/Safe-DS/Library/issues/112)
* rename `shuffle` to `shuffle_rows` ([#125](https://github.com/Safe-DS/Library/issues/125)) ([ea21928](https://github.com/Safe-DS/Library/commit/ea219285e869d0362339f8b87c310096cc001793))
* rename `slice` to `slice_rows` ([#126](https://github.com/Safe-DS/Library/issues/126)) ([20d21c2](https://github.com/Safe-DS/Library/commit/20d21c2fed8f85cfdcb6480b9f1f96ebafab64f9))
* rename `TableSchema` to `Schema` ([#133](https://github.com/Safe-DS/Library/issues/133)) ([1419d25](https://github.com/Safe-DS/Library/commit/1419d25113a28ed8ab76345a047eaf7dd4a3feb1))

## [0.7.0](https://github.com/Safe-DS/Library/compare/v0.6.0...v0.7.0) (2023-03-29)


### Features

* `sort_rows` of a `Table` ([#104](https://github.com/Safe-DS/Library/issues/104)) ([20aaf5e](https://github.com/Safe-DS/Library/commit/20aaf5eb276a0c756bb5414d4b268894d58a47e6)), closes [#14](https://github.com/Safe-DS/Library/issues/14)
* add `_file` suffix to methods interacting with files ([#103](https://github.com/Safe-DS/Library/issues/103)) ([ec011e4](https://github.com/Safe-DS/Library/commit/ec011e47d8a595ac6aa1c40d911b1b3da8cf5bd4))
* improve transformers for tabular data ([#108](https://github.com/Safe-DS/Library/issues/108)) ([b18a06d](https://github.com/Safe-DS/Library/commit/b18a06dce090a1bb9b6e3c858b83cd8b6277e280)), closes [#61](https://github.com/Safe-DS/Library/issues/61) [#90](https://github.com/Safe-DS/Library/issues/90)
* remove `OrdinalEncoder` ([#107](https://github.com/Safe-DS/Library/issues/107)) ([b92bba5](https://github.com/Safe-DS/Library/commit/b92bba551146586d510da03cc581037dc4c4c05e)), closes [#61](https://github.com/Safe-DS/Library/issues/61)
* specify features and target when creating a `TaggedTable` ([#114](https://github.com/Safe-DS/Library/issues/114)) ([95e1fc7](https://github.com/Safe-DS/Library/commit/95e1fc7b58dedda18f7fda43c9f6c45a57695f53)), closes [#27](https://github.com/Safe-DS/Library/issues/27)
* swap `name` and `data` parameters of `Column` ([#105](https://github.com/Safe-DS/Library/issues/105)) ([c2f8da5](https://github.com/Safe-DS/Library/commit/c2f8da537d1857bf89ec4417c1ba4f09ce5b8d49))

## [0.6.0](https://github.com/Safe-DS/Library/compare/v0.5.0...v0.6.0) (2023-03-27)


### Features

* allow calling `correlation_heatmap` with non-numerical columns ([#92](https://github.com/Safe-DS/Library/issues/92)) ([b960214](https://github.com/Safe-DS/Library/commit/b96021421f734fb7ca1b74e245a26b9997487292)), closes [#89](https://github.com/Safe-DS/Library/issues/89)
* function to drop columns with non-numerical values from `Table` ([#96](https://github.com/Safe-DS/Library/issues/96)) ([8f14d65](https://github.com/Safe-DS/Library/commit/8f14d65611cd9a1d6c75ae2769a4e5551c42b766)), closes [#13](https://github.com/Safe-DS/Library/issues/13)
* function to drop columns/rows with missing values ([#97](https://github.com/Safe-DS/Library/issues/97)) ([05d771c](https://github.com/Safe-DS/Library/commit/05d771c7fe9c0ea12ba7482a7ec5af197a450fce)), closes [#10](https://github.com/Safe-DS/Library/issues/10)
* remove `list_columns_with_XY` methods from `Table` ([#100](https://github.com/Safe-DS/Library/issues/100)) ([a0c56ad](https://github.com/Safe-DS/Library/commit/a0c56ad1671bd4388356dd952b398efc31fd8796)), closes [#94](https://github.com/Safe-DS/Library/issues/94)
* rename `keep_columns` to `keep_only_columns` ([#99](https://github.com/Safe-DS/Library/issues/99)) ([de42169](https://github.com/Safe-DS/Library/commit/de42169f6acde3d96985df24dc7f8213d96d2a4d))
* rename `remove_outliers` to `drop_rows_with_outliers` ([#95](https://github.com/Safe-DS/Library/issues/95)) ([7bad2e3](https://github.com/Safe-DS/Library/commit/7bad2e3e1b11fe45ed1fc408fa6289dfb5f301cb)), closes [#93](https://github.com/Safe-DS/Library/issues/93)
* return new model when calling `fit` ([#91](https://github.com/Safe-DS/Library/issues/91)) ([165c97c](https://github.com/Safe-DS/Library/commit/165c97c107aa52fddb6951c7092f2dccb164c64d)), closes [#69](https://github.com/Safe-DS/Library/issues/69)


### Bug Fixes

* handling of missing values when dropping rows with outliers ([#101](https://github.com/Safe-DS/Library/issues/101)) ([0a5e853](https://github.com/Safe-DS/Library/commit/0a5e853482ddeda147d5d6ff45e96166cfbfb1af)), closes [#7](https://github.com/Safe-DS/Library/issues/7)

## [0.5.0](https://github.com/Safe-DS/Library/compare/v0.4.0...v0.5.0) (2023-03-26)


### Features

* move plotting methods into `Column` and `Table` classes ([#88](https://github.com/Safe-DS/Library/issues/88)) ([5ec6189](https://github.com/Safe-DS/Library/commit/5ec6189a807092b00d38620403549c96a02164a5)), closes [#62](https://github.com/Safe-DS/Library/issues/62)

## [0.4.0](https://github.com/Safe-DS/Library/compare/v0.3.0...v0.4.0) (2023-03-26)


### Features

* better names for properties of `TaggedTable` ([#74](https://github.com/Safe-DS/Library/issues/74)) ([fee398b](https://github.com/Safe-DS/Library/commit/fee398b66cb9ae9e6675f455a8db31f271bfd207))
* change the name of a `Column` ([#76](https://github.com/Safe-DS/Library/issues/76)) ([ec539eb](https://github.com/Safe-DS/Library/commit/ec539eb6685d99183a35a138d1f345aaf6ae4c78))
* metrics as methods of models ([#77](https://github.com/Safe-DS/Library/issues/77)) ([bc63693](https://github.com/Safe-DS/Library/commit/bc636934a708b4a74aafed73fe4be75a7a36ebc4)), closes [#64](https://github.com/Safe-DS/Library/issues/64)
* optionally pass type to column ([#79](https://github.com/Safe-DS/Library/issues/79)) ([64aa429](https://github.com/Safe-DS/Library/commit/64aa4293bdf035fe4f9a78b0b895c07f022ced3a)), closes [#78](https://github.com/Safe-DS/Library/issues/78)
* remove `target_name` parameter of `predict` ([#70](https://github.com/Safe-DS/Library/issues/70)) ([b513454](https://github.com/Safe-DS/Library/commit/b513454c294f8ca03fbffa2b6f89a87e7d6fb9c6))
* rename `tagged_table` parameter of `fit` to `training_set` ([#71](https://github.com/Safe-DS/Library/issues/71)) ([8655521](https://github.com/Safe-DS/Library/commit/8655521bebbca2da9c91e2db7a837d4869a1d527))
* return `TaggedTable` from `predict` ([#73](https://github.com/Safe-DS/Library/issues/73)) ([5d5f5a6](https://github.com/Safe-DS/Library/commit/5d5f5a69d7e4def34ab09494511ae6ad6a62d60b))

## [0.3.0](https://github.com/Safe-DS/Library/compare/v0.2.0...v0.3.0) (2023-03-24)


### Features

* make `Column` and `Row` iterable ([#55](https://github.com/Safe-DS/Library/issues/55)) ([74eea1f](https://github.com/Safe-DS/Library/commit/74eea1f995d03732d14da16d4393e1d61ad33625)), closes [#47](https://github.com/Safe-DS/Library/issues/47)


### Bug Fixes

* "UserWarning: X has feature names" when predicting ([#53](https://github.com/Safe-DS/Library/issues/53)) ([74b0753](https://github.com/Safe-DS/Library/commit/74b07536f418732025f10cd6dc048cb61fab6cc5)), closes [#51](https://github.com/Safe-DS/Library/issues/51)
