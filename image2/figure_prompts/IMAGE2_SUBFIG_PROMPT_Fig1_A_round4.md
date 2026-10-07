# Figure 1A image2 重绘提示词：Norman--Weissman within-screen identity-held-out prediction contract

## 科学问题与总图作用

该子图回答一个明确的问题：在 Norman--Weissman Perturb-seq 单一公开 screen 内，当完整 perturbation identity 从拟合身份集合中留出、但查询 cell 的 gemgroup assay label 已在 observation metadata 中可用时，512-gene control-relative response prediction 的输入、训练聚合、锁定分割、推理评分和排除变量如何连接。它是整篇论文的模型机制与数据契约图，位于正文 Figure 1 的左上至右下阅读入口，读者应先从真实 perturbation identity 和 metadata 进入，再沿着训练分支看到 identity-by-gemgroup mean、ridge fit 和 locked identity split，最后沿推理分支到 cell-level 512-gene response score。图只表达已经在代码和结果中核验的结构关系，不表达性能数字、显著性、置信区间、样本量比较或跨 archive transfer。相邻定量图承担 RMSE、Pearson、Spearman 和消融数字；本图只解释这些数字的输入与 estimand，避免读者把 endpoint expression 当作 predictor 或把 gemgroup 当成生物机制。

## 期刊版面、画布和坐标布局

目标期刊是 Nature Communications，最终以双栏宽度约 180 mm 的横向 Figure 1 使用，建议画布 3600×2100 像素、300 dpi、白色背景，四周安全边距至少 120 px。采用 2 行 4 列的不规则网格：左上 x=140--930,y=180--850 为 identity input；左下 x=140--930,y=1120--1790 为 metadata and excluded endpoint audit；中央上 x=1040--1900,y=180--850 为 representation fusion；中央下 x=1040--1900,y=1120--1790 为 training aggregation and control reference；右上 x=2010--2870,y=180--850 为 identity-held-out split boundary；右下 x=2010--3460,y=1120--1790 为 ridge fitting、cell-level inference 和 512-gene output。网格之间保留 90--130 px 空隙，主体模块大小相近但 split boundary 与输出模块略大，避免四个空盒子组成的低信息量图。所有模块左对齐或中心对齐，箭头只沿水平或垂直方向走，不穿过文字，不交叉，不出现回头箭头。图内不放独立总标题，不放海报式长句；只保留 panel label “a”、短模块名和必要维度。

## 真实输入对象和逐字标签

左上 identity 输入必须显示三个并列但同属 identity contract 的真实块，逐字写为 “512 identity-gene indicators”, “256 Reactome scores” 和 “102 component indicators”。用小型多热向量、路径节点和 component token 三种不同但一致的视觉形式表示，明确它们都由 perturbation identity/guide annotation 产生；旁边放一个短标签 “complete perturbation identity p”，不能写成 measured expression。三个块汇入中央上 “870-dimensional identity representation f_p”。左下 metadata 区显示一个单独的八格 one-hot 小条，逐字写 “observed gemgroup b_i (8-level assay metadata)” 和 “available at prediction request”。它必须以橙色或琥珀色表示技术 covariate，并通过细实线进入融合模块；不要把 gemgroup 画成 DNA、细胞核、pathway 或 biological state 图标。

左下还要显示一个灰色 excluded block，逐字列出 “endpoint expression”, “library size”, “guide count”, “response-derived statistics”，并用灰色虚线框加一个小的禁止符号表示 “excluded from input”。禁止把这些字段画成箭头进入 model。旁边可放 “measured after response / not available to predictor” 的短注释，但不放长段落。灰色 excluded block 的边框和箭头必须弱于主数据流，保证科学边界清楚而不抢视觉中心。

## 训练分支、控制参考与箭头关系

中央下的 training branch 必须在一个浅蓝色分组区域中显示两个并列过程。上方过程从 identity representation f_p 和 gemgroup b 进入 “identity-by-gemgroup cells I_{pb}” 小型分组矩阵，再进入 “mean response y-bar_{pb}” 模块；箭头起点严格是训练 identities 的 cell rows，终点严格是每个 identity--gemgroup supported group 的 mean response。旁边放短标签 “minimum support: at least 2 cells”，不能写成任意阈值或样本权重。下方过程显示绿色小圆点组成的 “frozen CONTROL reference” 和一个短公式 “y_i = x_i - x-bar_CONTROL”，并注明 “3,207 CONTROL cells; fixed before fitting”。这只表示 target centering，不得画成 predictor branch 或把 control mean 连接到 gemgroup coefficient。控制参考箭头应进入 response-centering 模块，再以黑色短箭头连接到 mean response；不要暗示 control cells 参与 held-out identity membership。

## split boundary 与 ridge fit

右上 split boundary 用一个清晰的三段横条表示 “training identities 70%”, “validation identities 15%” 和 “confirmation identities 15%”，并在边界处放粗竖线和标签 “complete identity held out”。用紫色虚线将 validation-only alpha selection 指向 ridge fit，使用黑色实线将 training identity-by-gemgroup means 指向 ridge fit，使用绿色实线从 confirmation identities 指向 cell-level scoring。必须明确 split 的单位是 complete perturbation identity，而不是随机 cell；短标签写 “identity split, not random cell split”。不允许添加 external cohort、unseen batch、cross-cell-line 或 causal intervention arrows，因为这些关系没有在当前结果中核验。

右下模块按从左到右三层排列：第一层 “multi-output ridge fit” 与公式 “W_hat_alpha = argmin ||Y-XW||_F^2 + alpha||W||_F^2”；第二层 “cell-level query: (f_p, b_i)”；第三层 “predicted 512-gene control-relative response”. 在 query 到 output 之间画粗深灰箭头，明确每个 held-out cell 使用 identity representation 和 observed gemgroup one-hot。输出模块用 512 个细小彩色横线或矩阵格子表示 profile，但不写任何真实性能数字。输出旁边写 “RMSE / Pearson / Spearman scored on held-out cells” 作为短分析标签，表示评估位置，不制造数值。

## 视觉系统和可读性

使用 4--5 个主色并固定语义：深蓝表示 identity features and model representation，蓝绿色表示 response/target and control centering，橙色表示 observed gemgroup technical metadata，紫色表示 split/validation lock，灰色表示 excluded endpoint fields。箭头使用深灰或近黑，主数据流线宽 6 px、次级虚线 4 px，箭头头部明显且统一。所有文字使用 Arial/Helvetica 类无衬线字体；模块名 34--40 pt 等效，短标签 24--28 pt，维度数字 26--30 pt，公式 24--28 pt，最终双栏缩放后仍至少约 7.5 pt。标签首字母大小写统一，变量 $f_p$, $b_i$, $y_i$, $W$ 和下标必须清晰，不使用 Unicode 乱码。颜色在灰度打印中仍依靠边框、虚线和位置区分，避免相邻模块只靠相近色相区分。

## caption 分工与禁止事项

图内只留下模块名、维度、箭头语义和必要的 exclusions；数据来源、59,879 sampled cells、8 gemgroups、3,207 controls、5 seeds、具体指标值和每个面板结论全部放 caption 或正文。禁止独立总标题、长段落、装饰性图标、3D、发光、彩虹渐变、照片、无依据 pathway activation、因果箭头、external transfer claim、unseen-gemgroup coefficient、样本量或置信区间。image2 不得生成任何数字图表、性能曲线、误差线或显著性符号，也不得猜测模型参数或训练 epoch。不要加入论文没有的 decoder、attention、latent cell state、batch correction、drug dose 或 clinical outcome。

## 生成后检查与重试标准

生成后逐项核验：所有 512/256/102/8 维度是否逐字正确；identity、Reactome、component、gemgroup、control、split、ridge 和 output 的关系是否与正文一致；训练箭头是否从 training identity groups 指向 aggregation and fit；validation 只进入 alpha selection；confirmation 只进入 scoring；excluded endpoint fields 没有进入 model；所有箭头方向、线型和颜色语义是否明确；模块间是否过空或拥挤；在单栏和双栏缩放后文字是否可读；是否有裁切、乱码、错误符号、重复标签、错误的 biological causality 或外部迁移暗示。若任一标签、箭头或模块层级不准确，必须只修改 prompt 后重新调用 image2，保留原始 job 记录，不能通过 SVG、FigureSpec、Matplotlib、手工绘图或其他 fallback 冒充成功。只有 status=completed、nativeToolConfirmed=true、outputPath 存在且 PNG 文件头为 89 50 4e 47 0d 0a 1a 0a 才能作为正式图。
