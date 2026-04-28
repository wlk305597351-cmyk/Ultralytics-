# Neck 模块迁移设计文档

日期：2026-03-09（2026-03-10 增补）  
状态：设计已确认，待进入实现规划

## 1. 背景

当前仓库已经形成两套相对稳定的改进组织方式：

1. 运行时代码放在 `ultralytics/nn/extra_modules/<category>`
2. 配置文件放在 `ultralytics/cfg/models/improve/<category>`

本次需求以 `/root/code/project/ultralytics-yolo11` 中的 `yolo11-bifpn.yaml` 与 `yolo11-SDFM.yaml` 作为示例入口，但目标不是只落这两个模块。这份文档需要同时解决两件事：

1. 规范 neck 模块迁移到本仓库的通用落地方式
2. 给后续更多 neck 改进模块迁移提供统一模板

因此，这不是单个 YAML 的设计，而是一个可复用的 neck 模块迁移规范。

## 2. 目标

1. 为本仓库正式建立 `neck` 这一改进分类
2. 将首批 neck 模块所需的最小运行时依赖迁入本仓库
3. 在 `improve/neck` 下建立与现有改进目录风格一致的模型子目录结构
4. 为 `yolo11`、`yolo12`、`yolo26`、`yolov8`、`yolov10n` 提供首批检测版 neck YAML，并要求按家族补齐
5. 约束后续 neck 模块迁移时的目录、命名、注册和验证流程

## 3. 非目标

1. 不做“把源仓库 neck 相关代码整包搬过来”的粗放迁移
2. 不在首轮实现中顺带扩展到 `seg`、`pose`、`obb`
3. 不把 neck 模块继续塞进已有的 `block` 或 `featurefusion` 分类
4. 不为了省事重新堆一个大而杂的 `block.py`

## 4. 核心设计结论

### 4.1 新建运行时代码分类：`extra_modules/neck`

后续 neck 相关模块统一放到：

- `ultralytics/nn/extra_modules/neck`

这样做的原因：

1. neck 结构改进模块不应继续混在 `block`、`featurefusion` 等历史杂项里
2. 后续迁移更多 neck 模块时可以保持统一入口
3. 目录语义和 `improve/neck` 一一对应，后续维护成本更低

### 4.2 新建配置分类：`improve/neck`

neck 改进 YAML 统一放到：

- `ultralytics/cfg/models/improve/neck`

目录层级参考当前仓库的 `improve/downsample`，首批建立：

- `ultralytics/cfg/models/improve/neck/yolo11`
- `ultralytics/cfg/models/improve/neck/yolo12`
- `ultralytics/cfg/models/improve/neck/yolo26`
- `ultralytics/cfg/models/improve/neck/yolov8`
- `ultralytics/cfg/models/improve/neck/yolov10n`

其中 `yolov10n` 的目录名刻意对齐 `improve/downsample` 的现有习惯，不使用 `yolov10`。

### 4.3 首轮范围：先做检测版，且按家族补齐

首轮仅生成检测版 neck YAML，不扩到其他任务头。

对任意 neck 模块 `<NeckName>`，目标文件模式统一为：

- `ultralytics/cfg/models/improve/neck/yolo11/yolo11-<NeckName>.yaml`
- `ultralytics/cfg/models/improve/neck/yolo12/yolo12-<NeckName>.yaml`
- `ultralytics/cfg/models/improve/neck/yolo26/yolo26-<NeckName>.yaml`
- `ultralytics/cfg/models/improve/neck/yolov8/yolov8-<NeckName>.yaml`
- `ultralytics/cfg/models/improve/neck/yolov10n/yolov10n-<NeckName>.yaml`

上述文件均落在 `improve/neck/<model>/` 下，不再落到 `ultralytics/cfg/models/<version>/` 根目录。

`BiFPN`、`SDFM` 只是首批示例，不是规则的上限。

### 4.4 依赖迁移原则：只迁必要能力

首轮实现只迁目标 neck 模块真正需要的运行时能力，不连带迁移源仓库中与本次需求无关的 neck 变体或辅助模块。

任意 `<NeckName>` 的最小依赖链为：

1. `ultralytics/nn/extra_modules/neck/<NeckName>.py` 运行时模块
2. `ultralytics/nn/extra_modules/neck/__init__.py` 与 `ultralytics/nn/extra_modules/__init__.py` 导出注册
3. `ultralytics/nn/tasks.py` 中与 `<NeckName>` 对应的 `parse_model` 解析逻辑

源实现中存在但目标 YAML 未依赖的分支一律不迁，只有后续某个 neck YAML 明确依赖时再单独扩展。

## 5. 运行时代码设计

### 5.1 neck 模块文件组织规则

`extra_modules/neck` 采用“一个 neck 模块一份独立文件”的默认策略。

以本次 `BiFPN` 为例：

- `ultralytics/nn/extra_modules/neck/BiFPN.py`

后续 neck 模块按同样规则扩展，例如：

- `ultralytics/nn/extra_modules/neck/AFPN.py`
- `ultralytics/nn/extra_modules/neck/ASFF.py`
- `ultralytics/nn/extra_modules/neck/GoldNeck.py`

这样做的好处：

1. 每个 neck 模块的依赖边界更清楚
2. 避免未来再出现一个越来越大的“杂糅文件”
3. review、回归和后续替换都更容易

如果某个 neck 模块自带若干强耦合的内部 helper class，默认和主类放在同一个文件内，只有在确认可复用时再拆分。

### 5.2 neck 运行时接口通用约束

任意 neck 模块 `<NeckName>` 的运行时实现都应遵守：

1. 只实现目标 YAML 实际使用的最小行为
2. 输入、输出通道与 `parse_model` 推导规则保持一致
3. 模块文件固定落在 `ultralytics/nn/extra_modules/neck/<NeckName>.py`
4. 模块必须通过 `extra_modules/neck/__init__.py` 与 `extra_modules/__init__.py` 对外导出
5. 不把新 neck 模块继续塞回历史 `block.py` 杂项集合

### 5.3 示例（非强绑定）

以下仅为通用规则的示例，不构成模块白名单：

1. `BiFPN`：以 `Fusion` 为核心，通道规则受 `fusion_mode` 影响
2. `SDFM`：两路同尺度输入，输出通道与融合后单路特征一致

## 6. `parse_model` 设计

`ultralytics/nn/tasks.py` 需要把 neck 模块当作正式解析对象处理，而不是依赖 YAML 写法碰巧绕过去。

### 6.1 通用 `parse_model` 解析规则

对任意 `<NeckName>` 的解析逻辑应满足：

1. 从 YAML 的 `from`、`args` 还原运行时真实语义
2. 显式计算输入通道列表 `c1`
3. 明确给出输出通道 `c2` 推导规则
4. 将构造函数参数改写为运行时模块实际需要的签名

### 6.2 常见融合模式（示例）

1. 多分支可变模式融合：如 `Fusion`，`c2` 可能随模式变化（例如 `concat` 与非 `concat`）
2. 双分支定向融合：如 `SDFM`/`CGAFusion`，`c2` 通常按某一路输入对齐

### 6.3 后续 neck 模块的注册规则

后续新增 neck 模块时，遵守以下原则：

1. 只要通道推导有特殊性，就为它单独加解析分支
2. 只有在构造参数和输出通道规则已经完全匹配现有分支时，才复用现有解析逻辑
3. 不新增“杂项大集合”式注册分支，避免后续越来越难维护

## 7. YAML 设计规则

### 7.1 基础模型来源

每个 neck YAML 都必须从本仓库自己的基础模型 YAML 派生，而不是直接照搬外部仓库对应文件。

首批模型映射关系为：

- `yolo11` -> `ultralytics/cfg/models/11/yolo11.yaml`
- `yolo12` -> `ultralytics/cfg/models/12/yolo12.yaml`
- `yolo26` -> `ultralytics/cfg/models/26/yolo26.yaml`
- `yolov8` -> `ultralytics/cfg/models/v8/yolov8.yaml`
- `yolov10n` -> `ultralytics/cfg/models/v10/yolov10n.yaml`

### 7.2 首轮 YAML 改动范围

首轮 neck YAML 改动只允许聚焦在 neck/head 融合路径上，且所有文件统一落在 `ultralytics/cfg/models/improve/neck/<model>/`。

通用改动约束：

1. backbone 尽量保持基础模型原状
2. 按 `<NeckName>` 的融合逻辑修改 neck/head，不改无关路径
3. 若模块依赖全局参数（如 `fusion_mode`、`head_channel`），参数必须显式声明在 YAML 顶部
4. `Detect`/`v10Detect` 的输入层索引必须与新增 neck 节点一一对应

如果某个模型家族基础结构不同，不能机械复制 `yolo11` 的 neck 写法，而应以该家族自己的基础 YAML 做等价改写。

### 7.3 命名规则

统一命名约束如下：

1. 运行时文件名：`<NeckName>.py`
2. improve YAML 文件名：`<model>-<NeckName>.yaml`
3. 目录分类名固定为 `neck`

示例：

- `BiFPN.py` -> `yolo11-BiFPN.yaml`
- `SDFM.py` -> `yolo11-SDFM.yaml`
- `AFPN.py` -> `yolo12-AFPN.yaml`

如果运行时类名比较泛，例如 `Fusion`，文件名仍应优先体现 neck 家族语义，例如 `BiFPN.py`，这样后续查找更清晰。

## 8. 面向未来 neck 迁移的通用模板

这份文档后续要直接服务更多 neck 模块迁移，因此每次新增 neck 模块时，都应按以下模板执行。

### 8.1 第一步：判断是否属于 neck

满足以下任一主要职责时，可归入 `neck`：

1. 多尺度特征融合
2. 自顶向下或自底向上的路径聚合
3. backbone 输出到检测头输入之间的特征路由重构
4. 特征金字塔阶段之间的 neck 侧增强与重排

如果模块本质上是下采样、上采样、预处理、注意力或 backbone block，不要强行归到 `neck`。

### 8.2 第二步：先做最小依赖分析

迁移前必须先明确：

1. 哪个源 YAML 能证明该模块的实际用法
2. 该 YAML 实际依赖哪些运行时类
3. 哪些 helper class 在当前仓库不存在
4. 是否需要 `parse_model` 特殊分支

只迁真正被目标 YAML 用到的部分。

### 8.3 第三步：默认一模块一文件

新 neck 模块默认放在：

- `ultralytics/nn/extra_modules/neck/<ModuleName>.py`

除非两个模块就是同一个实现家族的不可拆分组成，否则不要提前做抽象合并。

### 8.4 第四步：按模型家族补 improve YAML

默认首轮覆盖以下检测模型家族：

- `yolo11`
- `yolo12`
- `yolo26`
- `yolov8`
- `yolov10n`

如果某个家族无法兼容，允许跳过，但必须在实现说明中明确写出原因，而不是悄悄少一个文件。

### 8.5 第五步：保持命名稳定

后续所有 neck 迁移都沿用以下稳定约束：

1. 代码文件名体现 neck 家族，而不是泛化为难以辨认的工具名
2. YAML 文件名与家族一一对应，保持 `<model>-<NeckName>.yaml`
3. 不因为单次迁移方便而破坏已有目录命名习惯

## 9. 首轮交付边界（通用）

基于本设计，任意 `<NeckName>` 首轮实现应交付：

1. `ultralytics/nn/extra_modules/neck/<NeckName>.py`
2. `ultralytics/nn/extra_modules/neck/__init__.py`
3. `ultralytics/nn/extra_modules/__init__.py` 中 neck 分类导出
4. `ultralytics/nn/tasks.py` 中 `<NeckName>` 对应解析逻辑
5. 以下检测 YAML：
    - `ultralytics/cfg/models/improve/neck/yolo11/yolo11-<NeckName>.yaml`
    - `ultralytics/cfg/models/improve/neck/yolo12/yolo12-<NeckName>.yaml`
    - `ultralytics/cfg/models/improve/neck/yolo26/yolo26-<NeckName>.yaml`
    - `ultralytics/cfg/models/improve/neck/yolov8/yolov8-<NeckName>.yaml`
    - `ultralytics/cfg/models/improve/neck/yolov10n/yolov10n-<NeckName>.yaml`

`BiFPN` 与 `SDFM` 可作为该交付模板的首批示例实现。

## 10. 验证规范

后续每一个 neck 模块迁移都按以下顺序验证：

1. 导入验证
    - 确认新模块能从 `ultralytics.nn.extra_modules` 正常导入
2. `parse_model` 冒烟
    - 确认新增 YAML 能被模型解析器成功加载
3. 家族覆盖检查
    - `yolo11`、`yolo12`、`yolo26`、`yolov8`、`yolov10n` 五个家族必须全部补齐
    - 不允许只迁 `yolo11` 而缺失其余家族文件
4. YAML 快速验证
    - 对每个新 YAML 单独做快速测试
    - 记录参数量、GFLOPs、stride
5. 异常分流
    - 若某个模型家族因基础结构差异导致无法直接适配，应明确记录原因，并决定修正或暂缓，而不是无说明跳过

## 11. 风险与规避

### 11.1 模型家族之间 neck 结构并不完全同构

`yolo11`、`yolo12`、`yolo26`、`yolov8`、`yolov10n` 的 neck 结构未必能直接互套。

规避方式：

每个 `<NeckName>` YAML 都从本仓库对应基础 YAML 推导，不复制 `yolo11` 的 neck 到其它家族。

### 11.2 源仓库常把多种融合分支写在一起

如果直接搬整段源代码，很容易把本次根本用不到的分支一起带进来。

规避方式：

先以最小实现满足当前 YAML，再按后续真实需求逐步扩展。

### 11.3 未来 neck 模块可能和其他分类边界模糊

某些模块可能同时带 attention、preprocess 或 fusion 属性。

规避方式：

按“主要职责”分类：如果核心是金字塔结构、路径聚合或多尺度路由重构，则归 `neck`；否则放回已有更准确的分类。

## 12. 下一步实现规划应覆盖的内容

基于本设计，下一阶段实现计划至少应明确：

1. 文件创建与注册顺序
2. `<NeckName>` 运行时代码的最小实现边界
3. 五个模型家族各自的 YAML 推导策略
4. 验证命令与失败回退策略
5. 如果某个家族需要结构特判，应如何记录并处理

## 13. 有意延后、不在本轮设计中拍板的事项

1. 后续 neck 模块是否同步补 `module_images`
2. `seg`、`pose`、`obb` 是否也要统一建立 `improve/neck` 变体
3. 当 neck 模块数量增多后，是否抽象公共解析 helper
