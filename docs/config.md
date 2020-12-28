# 配置文件使用方法

## `configs/configs.py`配置

该文件配置了：

- 依赖的路径，临时文件夹、输出文件夹的路径
  - `SA_HOME` 后续路径配置的公共前缀，是为了尽量减少绝对路径设置的。

  - `ALGO_BASE_DIRS` 算法相关的路径配置，其中“TmpDir”是程序运行时临时文件存放路径。如README所述，其中“JDC”是旋律提取算法的路径，“PopMusicHighlighter”是对比算法“highlighter”的路径。

  - `DATASET_BASE_DIRS` 数据集路径配置，其中“RWC”是RWC数据集目录位置，“LocalTemporary_Dataset”是预处理文件或缓存结果的存储位置。

  - `EVAL_RESULT_DIR`,`MODELS_DIR`,`VIEWER_DATA_DIR`,`PRED_DIR` 分别为评估结果、模型训练数据、可视化预测结果、标准预测结果文件的存储目录。

- 并行计算数量 `NUM_WORKERS`

- 评估指标名`METRIC_NAMES`、绘图中使用到的指标`PLOT_METRIC_FIELDS`、评估指标参数（检测窗口长度）`DETECTION_WINDOW`

- 日志记录器`logger`与日志等级`DEBUG`

## `configs/modelConfigs.py`配置

该文件配置了算法相关的参数，部分可修改的配置为：

- `USING_DATASET` 为算法使用的数据集，建议设置为`RWC_Popular_Dataset()`

- `SSM_USING_MELODY` 是否使用旋律线，建议设置为`True`
