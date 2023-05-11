## [0.12.0](https://github.com/Safe-DS/Stdlib/compare/v0.11.0...v0.12.0) (2023-05-11)


### Features

* add `learning_rate` to AdaBoost classifier and regressor. ([#251](https://github.com/Safe-DS/Stdlib/issues/251)) ([7f74440](https://github.com/Safe-DS/Stdlib/commit/7f744409c08fb465d59f1f04e2cac7ebed23f339)), closes [#167](https://github.com/Safe-DS/Stdlib/issues/167)
* add alpha parameter to `lasso_regression` ([#232](https://github.com/Safe-DS/Stdlib/issues/232)) ([b5050b9](https://github.com/Safe-DS/Stdlib/commit/b5050b91f17774fa5cf3fc80b51d3ea6c295481f)), closes [#163](https://github.com/Safe-DS/Stdlib/issues/163)
* add parameter `lasso_ratio` to `ElasticNetRegression` ([#237](https://github.com/Safe-DS/Stdlib/issues/237)) ([4a1a736](https://github.com/Safe-DS/Stdlib/commit/4a1a7367099125d2a072bf36686063de7180e8f0)), closes [#166](https://github.com/Safe-DS/Stdlib/issues/166)
* Add parameter `number_of_tree` to `RandomForest` classifier and regressor ([#230](https://github.com/Safe-DS/Stdlib/issues/230)) ([414336a](https://github.com/Safe-DS/Stdlib/commit/414336ac9752f961cab30545cbe51befbde50d21)), closes [#161](https://github.com/Safe-DS/Stdlib/issues/161)
* Added `Table.plot_boxplots` to plot a boxplot for each numerical column in the table ([#254](https://github.com/Safe-DS/Stdlib/issues/254)) ([0203a0c](https://github.com/Safe-DS/Stdlib/commit/0203a0c977184cdee1769d317fcb1f7cb5c644f3)), closes [#156](https://github.com/Safe-DS/Stdlib/issues/156) [#239](https://github.com/Safe-DS/Stdlib/issues/239)
* Added `Table.plot_histograms` to plot a histogram for each column in the table ([#252](https://github.com/Safe-DS/Stdlib/issues/252)) ([e27d410](https://github.com/Safe-DS/Stdlib/commit/e27d410085ebaf9ab98069a5b175d800259d95a3)), closes [#157](https://github.com/Safe-DS/Stdlib/issues/157)
* Added `Table.transform_table` method which returns the transformed Table ([#229](https://github.com/Safe-DS/Stdlib/issues/229)) ([0a9ce72](https://github.com/Safe-DS/Stdlib/commit/0a9ce72ba2101f99fea43dcd43b1f498dbb8e558)), closes [#110](https://github.com/Safe-DS/Stdlib/issues/110)
* Added alpha parameter to `RidgeRegression` ([#231](https://github.com/Safe-DS/Stdlib/issues/231)) ([1ddc948](https://github.com/Safe-DS/Stdlib/commit/1ddc948aac5f153f649c3869b99184c2c1d96d9f)), closes [#164](https://github.com/Safe-DS/Stdlib/issues/164)
* Added Column#transform ([#270](https://github.com/Safe-DS/Stdlib/issues/270)) ([40fb756](https://github.com/Safe-DS/Stdlib/commit/40fb7566307b4c015f3acae7bb94f4e937977e07)), closes [#255](https://github.com/Safe-DS/Stdlib/issues/255)
* Added method `Table.inverse_transform_table` which returns the original table ([#227](https://github.com/Safe-DS/Stdlib/issues/227)) ([846bf23](https://github.com/Safe-DS/Stdlib/commit/846bf233235b2cdaf9bbd00cacb89ea44e94011b)), closes [#111](https://github.com/Safe-DS/Stdlib/issues/111)
* Added parameter `c` to `SupportVectorMachines` ([#267](https://github.com/Safe-DS/Stdlib/issues/267)) ([a88eb8b](https://github.com/Safe-DS/Stdlib/commit/a88eb8b8c3f67e8485ce2847c4923a2cf0506f68)), closes [#169](https://github.com/Safe-DS/Stdlib/issues/169)
* Added parameter `maximum_number_of_learner` and `learner` to `AdaBoost` ([#269](https://github.com/Safe-DS/Stdlib/issues/269)) ([bb5a07e](https://github.com/Safe-DS/Stdlib/commit/bb5a07e17b6563d394d79b62349633791675346f)), closes [#171](https://github.com/Safe-DS/Stdlib/issues/171) [#173](https://github.com/Safe-DS/Stdlib/issues/173)
* Added parameter `number_of_trees` to `GradientBoosting` ([#268](https://github.com/Safe-DS/Stdlib/issues/268)) ([766f2ff](https://github.com/Safe-DS/Stdlib/commit/766f2ff2a6d68098be3e858ad12bf9e509e5f192)), closes [#170](https://github.com/Safe-DS/Stdlib/issues/170)
* Allow arguments of type pathlib.Path for file I/O methods ([#228](https://github.com/Safe-DS/Stdlib/issues/228)) ([2b58c82](https://github.com/Safe-DS/Stdlib/commit/2b58c82f50ce88b4778f3c82108f5d5f474fdfa9)), closes [#146](https://github.com/Safe-DS/Stdlib/issues/146)
* convert `Schema` to `dict` and format it nicely in a notebook ([#244](https://github.com/Safe-DS/Stdlib/issues/244)) ([ad1cac5](https://github.com/Safe-DS/Stdlib/commit/ad1cac5198709d0a78019787251ba2aed913cf55)), closes [#151](https://github.com/Safe-DS/Stdlib/issues/151)
* Convert between Excel file and `Table` ([#233](https://github.com/Safe-DS/Stdlib/issues/233)) ([0d7a998](https://github.com/Safe-DS/Stdlib/commit/0d7a998f9e660f47147f7eaa6ebb8119c09188ac)), closes [#138](https://github.com/Safe-DS/Stdlib/issues/138) [#139](https://github.com/Safe-DS/Stdlib/issues/139)
* convert containers for tabular data to HTML ([#243](https://github.com/Safe-DS/Stdlib/issues/243)) ([683c279](https://github.com/Safe-DS/Stdlib/commit/683c2793f053f5d0572e723b35db383aa00ddc44)), closes [#140](https://github.com/Safe-DS/Stdlib/issues/140)
* make `Column` a subclass of `Sequence` ([#245](https://github.com/Safe-DS/Stdlib/issues/245)) ([a35b943](https://github.com/Safe-DS/Stdlib/commit/a35b943a126b28500499f5d7da1bccee10d98ff3))
* mark optional hyperparameters as keyword only ([#296](https://github.com/Safe-DS/Stdlib/issues/296)) ([44a41eb](https://github.com/Safe-DS/Stdlib/commit/44a41eb205ad0f69f01564ab318e53873bb902c4)), closes [#278](https://github.com/Safe-DS/Stdlib/issues/278)
* move exceptions back to common package ([#295](https://github.com/Safe-DS/Stdlib/issues/295)) ([a91172c](https://github.com/Safe-DS/Stdlib/commit/a91172c0f21ea9934cedbe9fd749eb4ff7929394)), closes [#177](https://github.com/Safe-DS/Stdlib/issues/177) [#262](https://github.com/Safe-DS/Stdlib/issues/262)
* precision metric for classification ([#272](https://github.com/Safe-DS/Stdlib/issues/272)) ([5adadad](https://github.com/Safe-DS/Stdlib/commit/5adadadf6ab185b4d8864b7859d7cc036a055a6d)), closes [#185](https://github.com/Safe-DS/Stdlib/issues/185)
* Raise error if an untagged table is used instead of a `TaggedTable` ([#234](https://github.com/Safe-DS/Stdlib/issues/234)) ([8eea3dd](https://github.com/Safe-DS/Stdlib/commit/8eea3dd31dab49b4d9371f61f02ace9fdca25394)), closes [#192](https://github.com/Safe-DS/Stdlib/issues/192)
* recall and F1-score metrics for classification ([#277](https://github.com/Safe-DS/Stdlib/issues/277)) ([2cf93cc](https://github.com/Safe-DS/Stdlib/commit/2cf93cc7181ad69991055dd0e49035a785105356)), closes [#187](https://github.com/Safe-DS/Stdlib/issues/187) [#186](https://github.com/Safe-DS/Stdlib/issues/186)
* replace prefix `n` with `number_of` ([#250](https://github.com/Safe-DS/Stdlib/issues/250)) ([f4f44a6](https://github.com/Safe-DS/Stdlib/commit/f4f44a6b8d5f8ee795673b11c5f00e3ebb1b1b39)), closes [#171](https://github.com/Safe-DS/Stdlib/issues/171)
* set `alpha` parameter for regularization of `ElasticNetRegression` ([#238](https://github.com/Safe-DS/Stdlib/issues/238)) ([e642d1d](https://github.com/Safe-DS/Stdlib/commit/e642d1d49c5b21240fa5bbbde48e80d5b7743ff1)), closes [#165](https://github.com/Safe-DS/Stdlib/issues/165)
* Set `column_names` in `fit` methods of table transformers to be required ([#225](https://github.com/Safe-DS/Stdlib/issues/225)) ([2856296](https://github.com/Safe-DS/Stdlib/commit/2856296fb7228e8d4adebceb86e22ecaeb609ad9)), closes [#179](https://github.com/Safe-DS/Stdlib/issues/179)
* set learning rate of Gradient Boosting models ([#253](https://github.com/Safe-DS/Stdlib/issues/253)) ([9ffaf55](https://github.com/Safe-DS/Stdlib/commit/9ffaf55a97333bb2edce2f2c9c66650a9724ca60)), closes [#168](https://github.com/Safe-DS/Stdlib/issues/168)
* Support vector machine for regression and for classification ([#236](https://github.com/Safe-DS/Stdlib/issues/236)) ([7f6c3bd](https://github.com/Safe-DS/Stdlib/commit/7f6c3bd9fba670a487d3ef96d281f3904a8974a7)), closes [#154](https://github.com/Safe-DS/Stdlib/issues/154)
* usable constructor for `Table` ([#294](https://github.com/Safe-DS/Stdlib/issues/294)) ([56a1fc4](https://github.com/Safe-DS/Stdlib/commit/56a1fc4450ba77877b6b29467c0e1d11dd200f9d)), closes [#266](https://github.com/Safe-DS/Stdlib/issues/266)
* usable constructor for `TaggedTable` ([#299](https://github.com/Safe-DS/Stdlib/issues/299)) ([01c3ad9](https://github.com/Safe-DS/Stdlib/commit/01c3ad9564a35f31744a30862ae1a533ef5adf6b)), closes [#293](https://github.com/Safe-DS/Stdlib/issues/293)


### Bug Fixes

* OneHotEncoder no longer creates duplicate column names ([#271](https://github.com/Safe-DS/Stdlib/issues/271)) ([f604666](https://github.com/Safe-DS/Stdlib/commit/f604666305d38d3a01696ea7ca60056ce7f78245)), closes [#201](https://github.com/Safe-DS/Stdlib/issues/201)
* selectively ignore one warning instead of all warnings ([#235](https://github.com/Safe-DS/Stdlib/issues/235)) ([3aad07d](https://github.com/Safe-DS/Stdlib/commit/3aad07ddcc0da42e1dab2eed49fc41433a876765))

## [0.11.0](https://github.com/Safe-DS/Stdlib/compare/v0.10.0...v0.11.0) (2023-04-21)


### Features

* `OneHotEncoder.inverse_transform` now maintains the column order from the original table ([#195](https://github.com/Safe-DS/Stdlib/issues/195)) ([3ec0041](https://github.com/Safe-DS/Stdlib/commit/3ec0041669ffe97640f96db345f3f43720d5c3f7)), closes [#109](https://github.com/Safe-DS/Stdlib/issues/109)
* add `plot_` prefix back to plotting methods ([#212](https://github.com/Safe-DS/Stdlib/issues/212)) ([e50c3b0](https://github.com/Safe-DS/Stdlib/commit/e50c3b0118825e33eef0e2a7073673603e316ee5)), closes [#211](https://github.com/Safe-DS/Stdlib/issues/211)
* adjust `Column`, `Schema` and `Table` to changes in `Row` ([#216](https://github.com/Safe-DS/Stdlib/issues/216)) ([ca3eebb](https://github.com/Safe-DS/Stdlib/commit/ca3eebbe2166f08d76cdcb89a012518ae8ff8e4e))
* back `Row` by a `polars.DataFrame` ([#214](https://github.com/Safe-DS/Stdlib/issues/214)) ([62ca34d](https://github.com/Safe-DS/Stdlib/commit/62ca34dd399da8b4e34b89bad408311707b53f41)), closes [#196](https://github.com/Safe-DS/Stdlib/issues/196) [#149](https://github.com/Safe-DS/Stdlib/issues/149)
* clean up `Row` class ([#215](https://github.com/Safe-DS/Stdlib/issues/215)) ([b12fc68](https://github.com/Safe-DS/Stdlib/commit/b12fc68c9b914446c1ea3aca5dacfab969680ae8))
* convert between `Row` and `dict` ([#206](https://github.com/Safe-DS/Stdlib/issues/206)) ([e98b653](https://github.com/Safe-DS/Stdlib/commit/e98b6536a2c50e64772fc7aeb29c03c850ebd570)), closes [#204](https://github.com/Safe-DS/Stdlib/issues/204)
* convert between a `dict` and a `Table` ([#198](https://github.com/Safe-DS/Stdlib/issues/198)) ([2a5089e](https://github.com/Safe-DS/Stdlib/commit/2a5089e408a1eeb078aa77ce7c3e5ae8c4bc0201)), closes [#197](https://github.com/Safe-DS/Stdlib/issues/197)
* create column types for `polars` data types ([#208](https://github.com/Safe-DS/Stdlib/issues/208)) ([e18b362](https://github.com/Safe-DS/Stdlib/commit/e18b36250ac170e3364106ba1c59649e0b4aff21)), closes [#196](https://github.com/Safe-DS/Stdlib/issues/196)
* dataframe interchange protocol ([#200](https://github.com/Safe-DS/Stdlib/issues/200)) ([bea976a](https://github.com/Safe-DS/Stdlib/commit/bea976a72a28698a29145a3ad501d8af59e7e5d8)), closes [#199](https://github.com/Safe-DS/Stdlib/issues/199)
* move existing ML solutions into `safeds.ml.classical` package ([#213](https://github.com/Safe-DS/Stdlib/issues/213)) ([655f07f](https://github.com/Safe-DS/Stdlib/commit/655f07f7f8f02b8fc92b469daf15a2384a81534f)), closes [#210](https://github.com/Safe-DS/Stdlib/issues/210)


### Bug Fixes

* `table.keep_only_columns` now maps column names to correct data ([#194](https://github.com/Safe-DS/Stdlib/issues/194)) ([459ab75](https://github.com/Safe-DS/Stdlib/commit/459ab7570c7c7b79304f78cab4f6bff82fc026a3)), closes [#115](https://github.com/Safe-DS/Stdlib/issues/115)
* typo in type hint ([#184](https://github.com/Safe-DS/Stdlib/issues/184)) ([e79727d](https://github.com/Safe-DS/Stdlib/commit/e79727d5d91090bc5cd6d3a81acc2a1393b3e5eb)), closes [#180](https://github.com/Safe-DS/Stdlib/issues/180)

## [0.10.0](https://github.com/Safe-DS/Stdlib/compare/v0.9.0...v0.10.0) (2023-04-13)


### Features

* move exceptions into subpackages ([#177](https://github.com/Safe-DS/Stdlib/issues/177)) ([10b17fd](https://github.com/Safe-DS/Stdlib/commit/10b17fddca6518dd0d62da0a791c508659c994c4))

## [0.9.0](https://github.com/Safe-DS/Stdlib/compare/v0.8.0...v0.9.0) (2023-04-04)


### Features

* container for images ([#159](https://github.com/Safe-DS/Stdlib/issues/159)) ([ed7ae34](https://github.com/Safe-DS/Stdlib/commit/ed7ae341c4546ec32efe46e22dccc4d770126695)), closes [#158](https://github.com/Safe-DS/Stdlib/issues/158)
* improve error handling for `predict` ([#145](https://github.com/Safe-DS/Stdlib/issues/145)) ([a5ff11c](https://github.com/Safe-DS/Stdlib/commit/a5ff11c2795e8e814b6a6565a9a331e1662d39c6)), closes [#9](https://github.com/Safe-DS/Stdlib/issues/9)
* move `ImputerStrategy` to `safeds.data.tabular.typing` ([#174](https://github.com/Safe-DS/Stdlib/issues/174)) ([205c8e2](https://github.com/Safe-DS/Stdlib/commit/205c8e20ddcc76da57b895a23c0221da4dcf2508))
* rename `n_neighbors` to `number_of_neighbors` ([#162](https://github.com/Safe-DS/Stdlib/issues/162)) ([526b96e](https://github.com/Safe-DS/Stdlib/commit/526b96e3877299eb6bf6adea2882065fd29b52cf))


### Bug Fixes

* export `TableTransformer` and `InvertibleTableTransformer` ([#135](https://github.com/Safe-DS/Stdlib/issues/135)) ([81c3695](https://github.com/Safe-DS/Stdlib/commit/81c369556e8ca3bf800f843598efab29b0ac957b))

## [0.8.0](https://github.com/Safe-DS/Stdlib/compare/v0.7.0...v0.8.0) (2023-03-31)


### Features

* create empty `Table` without schema ([#128](https://github.com/Safe-DS/Stdlib/issues/128)) ([ddd3f59](https://github.com/Safe-DS/Stdlib/commit/ddd3f59cf4f0173327511593ea4dc0b5b938fce1)), closes [#127](https://github.com/Safe-DS/Stdlib/issues/127)
* improve `ColumnType`s ([#132](https://github.com/Safe-DS/Stdlib/issues/132)) ([1786a87](https://github.com/Safe-DS/Stdlib/commit/1786a872f9fe713b952e75c190245200082ac80d)), closes [#113](https://github.com/Safe-DS/Stdlib/issues/113)
* infer schema of row if not passed explicitly ([#134](https://github.com/Safe-DS/Stdlib/issues/134)) ([c5869bb](https://github.com/Safe-DS/Stdlib/commit/c5869bbc215d884c48726b3c8f6b3556763d986d)), closes [#15](https://github.com/Safe-DS/Stdlib/issues/15)
* new method `is_fitted` to check whether a model is fitted ([#130](https://github.com/Safe-DS/Stdlib/issues/130)) ([8e1c3ea](https://github.com/Safe-DS/Stdlib/commit/8e1c3ea22c3b422b65340f6fc25a87d0d7fb8869))
* new method `is_fitted` to check whether a transformer is fitted ([#131](https://github.com/Safe-DS/Stdlib/issues/131)) ([e20954f](https://github.com/Safe-DS/Stdlib/commit/e20954feb0f9191596aac93672b67361e1aa4ef8))
* rename `drop_XY` methods of `Table` to `remove_XY` ([#122](https://github.com/Safe-DS/Stdlib/issues/122)) ([98d76a4](https://github.com/Safe-DS/Stdlib/commit/98d76a46a56d4f78cb30d3ea8c4916b69f738674))
* rename `fit_transform` to `fit_and_transform` ([#119](https://github.com/Safe-DS/Stdlib/issues/119)) ([76a7112](https://github.com/Safe-DS/Stdlib/commit/76a71126b6ca21f9082dd2d3a2248bf65716b73f)), closes [#112](https://github.com/Safe-DS/Stdlib/issues/112)
* rename `shuffle` to `shuffle_rows` ([#125](https://github.com/Safe-DS/Stdlib/issues/125)) ([ea21928](https://github.com/Safe-DS/Stdlib/commit/ea219285e869d0362339f8b87c310096cc001793))
* rename `slice` to `slice_rows` ([#126](https://github.com/Safe-DS/Stdlib/issues/126)) ([20d21c2](https://github.com/Safe-DS/Stdlib/commit/20d21c2fed8f85cfdcb6480b9f1f96ebafab64f9))
* rename `TableSchema` to `Schema` ([#133](https://github.com/Safe-DS/Stdlib/issues/133)) ([1419d25](https://github.com/Safe-DS/Stdlib/commit/1419d25113a28ed8ab76345a047eaf7dd4a3feb1))

## [0.7.0](https://github.com/Safe-DS/Stdlib/compare/v0.6.0...v0.7.0) (2023-03-29)


### Features

* `sort_rows` of a `Table` ([#104](https://github.com/Safe-DS/Stdlib/issues/104)) ([20aaf5e](https://github.com/Safe-DS/Stdlib/commit/20aaf5eb276a0c756bb5414d4b268894d58a47e6)), closes [#14](https://github.com/Safe-DS/Stdlib/issues/14)
* add `_file` suffix to methods interacting with files ([#103](https://github.com/Safe-DS/Stdlib/issues/103)) ([ec011e4](https://github.com/Safe-DS/Stdlib/commit/ec011e47d8a595ac6aa1c40d911b1b3da8cf5bd4))
* improve transformers for tabular data ([#108](https://github.com/Safe-DS/Stdlib/issues/108)) ([b18a06d](https://github.com/Safe-DS/Stdlib/commit/b18a06dce090a1bb9b6e3c858b83cd8b6277e280)), closes [#61](https://github.com/Safe-DS/Stdlib/issues/61) [#90](https://github.com/Safe-DS/Stdlib/issues/90)
* remove `OrdinalEncoder` ([#107](https://github.com/Safe-DS/Stdlib/issues/107)) ([b92bba5](https://github.com/Safe-DS/Stdlib/commit/b92bba551146586d510da03cc581037dc4c4c05e)), closes [#61](https://github.com/Safe-DS/Stdlib/issues/61)
* specify features and target when creating a `TaggedTable` ([#114](https://github.com/Safe-DS/Stdlib/issues/114)) ([95e1fc7](https://github.com/Safe-DS/Stdlib/commit/95e1fc7b58dedda18f7fda43c9f6c45a57695f53)), closes [#27](https://github.com/Safe-DS/Stdlib/issues/27)
* swap `name` and `data` parameters of `Column` ([#105](https://github.com/Safe-DS/Stdlib/issues/105)) ([c2f8da5](https://github.com/Safe-DS/Stdlib/commit/c2f8da537d1857bf89ec4417c1ba4f09ce5b8d49))

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
* remove `target_name` parameter of `predict` ([#70](https://github.com/Safe-DS/Stdlib/issues/70)) ([b513454](https://github.com/Safe-DS/Stdlib/commit/b513454c294f8ca03fbffa2b6f89a87e7d6fb9c6))
* rename `tagged_table` parameter of `fit` to `training_set` ([#71](https://github.com/Safe-DS/Stdlib/issues/71)) ([8655521](https://github.com/Safe-DS/Stdlib/commit/8655521bebbca2da9c91e2db7a837d4869a1d527))
* return `TaggedTable` from `predict` ([#73](https://github.com/Safe-DS/Stdlib/issues/73)) ([5d5f5a6](https://github.com/Safe-DS/Stdlib/commit/5d5f5a69d7e4def34ab09494511ae6ad6a62d60b))

## [0.3.0](https://github.com/Safe-DS/Stdlib/compare/v0.2.0...v0.3.0) (2023-03-24)


### Features

* make `Column` and `Row` iterable ([#55](https://github.com/Safe-DS/Stdlib/issues/55)) ([74eea1f](https://github.com/Safe-DS/Stdlib/commit/74eea1f995d03732d14da16d4393e1d61ad33625)), closes [#47](https://github.com/Safe-DS/Stdlib/issues/47)


### Bug Fixes

* "UserWarning: X has feature names" when predicting ([#53](https://github.com/Safe-DS/Stdlib/issues/53)) ([74b0753](https://github.com/Safe-DS/Stdlib/commit/74b07536f418732025f10cd6dc048cb61fab6cc5)), closes [#51](https://github.com/Safe-DS/Stdlib/issues/51)
