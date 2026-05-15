# YOLO26 创新点集成方案

## 概述

将 YOLO26 的三大检测创新点集成到当前基于 YOLO11 的双模态(RGB+Thermal)目标检测项目中。

**核心原则：所有旧代码注释保留不删除，每个改动位置用 `[YOLO26]` 标记注释。**

| 创新点 | 说明 | 涉及文件 |
|--------|------|---------|
| **MuSGD 优化器** | 融合 Muon(Newton-Schulz正交化) + SGD，加速收敛 | trainer.py |
| **ProgLoss + E2ELoss** | 渐进式损失权重(0.8→0.1) + one2one topk=7→1 | loss.py, tasks.py |
| **DFL-free (reg_max=1)** | 移除分布焦点损失，直接L1坐标回归 | tasks.py(parse_model) + YAML |
| **STAL** | one2one 先选top7候选再筛到top1，小目标更容易匹配 | 包含在E2ELoss中 |

---

## 改动详情

### 改动1: `ultralytics/engine/trainer.py` — 集成 MuSGD 优化器

**现状**: 已有 `MuSGD` 代码(`optim/muon.py`)，但 trainer 未集成，仅支持 SGD/AdamW 等

**YOLO26 做法**:
- 参数分 4 组: `[weight_decay, no_decay, bias, muon_params]`
- `muon_params`: 所有 `ndim >= 2` 的权重使用 Muon 更新
- `auto` 模式且 iterations > 10000 时默认选 MuSGD
- MuSGD 同时执行 Muon + SGD 更新(momentum)

**改动** (`build_optimizer`方法，约760行):
```
g = [{}, {}, {}, {}]   # [YOLO26] 新增第4组
use_muon = name == "MuSGD"
# 遍历参数时: if param.ndim >= 2 and use_muon: g[3][fullname] = param
# auto模式: name = "MuSGD" if iterations > 10000
# 构建: MuSGD(params=g, muon=0.2, sgd=1.0)
```

---

### 改动2: `ultralytics/utils/loss.py` — 升级为 E2ELoss (ProgLoss + STAL)

**现状**: 使用 `E2EDetectLoss`(YOLOv10风格)，两个分支loss直接相加

**YOLO26 E2ELoss**:
- one2many: `tal_topk=10` (不变)
- one2one: `tal_topk=7, tal_topk2=1` ← **STAL**: 先选7个候选再筛到最佳
- **ProgLoss**: o2m 权重从 0.8 线性衰减到 0.1
  - 初期(0.8/0.2): 侧重 one2many，分类+粗定位优先
  - 后期(0.1/0.9): 侧重 one2one 精细定位，弥补无DFL的精度损失

**改动** (~1280行):
```
# [YOLO26] 旧的 E2EDetectLoss 注释保留
# class E2EDetectLoss: ...

# [YOLO26] 新增 E2ELoss 类
class E2ELoss:
    def __init__(self, model, loss_fn=v8DetectionLoss):
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.o2m = 0.8; self.o2o = 0.2; self.final_o2m = 0.1
    def update(self):
        self.o2m = max(1 - self.updates / (epochs - 1), 0) * (0.8 - 0.1) + 0.1
        self.o2o = max(1.0 - self.o2m, 0)
```

---

### 改动3: `ultralytics/nn/tasks.py` — parse_model + DetectionModel + 导入

**3a. 导入更新 (~85行)**:
```
# [YOLO26] 旧导入注释保留
# from ultralytics.utils.loss import (E2EDetectLoss, ...)
# [YOLO26] 新增 E2ELoss 导入
from ultralytics.utils.loss import (E2ELoss, ...)
```

**3b. parse_model 读 YAML 的 reg_max/end2end (~975行)**:
```
# [YOLO26] 旧代码注释保留
# nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
# [YOLO26] 新增: 从YAML顶层读取reg_max和end2end
nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
reg_max = d.get("reg_max", 16)
```

**3c. parse_model Detect head参数传递 (~1273行)**:
```
# [YOLO26] 旧代码注释保留
# args.append([ch[x] for x in f])
# [YOLO26] 显式传递reg_max和end2end
args.extend([reg_max, end2end, [ch[x] for x in f]])
```

**3d. DetectionModel.init_criterion() (~426行)**:
```
# [YOLO26] 旧代码注释保留
# return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
# [YOLO26] 使用 E2ELoss
return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
```

**3e. DetectionModel.loss() (~317行)**:
```
def loss(self, batch, preds=None):
    ...
    preds = self.forward(batch["img"]) if preds is None else preds
    # [YOLO26] 每个batch更新ProgLoss渐进权重
    if hasattr(self.criterion, "update"):
        self.criterion.update()
    return self.criterion(preds, batch)
```

---

### 改动4: `ultralytics/nn/modules/head.py` — 无需改动

现有 `Detect` 类已接受 `reg_max` 和 `end2end` 参数（见 `__init__` 签名）：
```python
def __init__(self, nc=80, reg_max=16, end2end=False, ch=())
```

`BboxLoss` 已支持 `reg_max=1`（跳过DFL，使用L1 regression）。

YAML中设置 `reg_max: 1` + `end2end: True` 后，通过改动3b/3c自动传递到Detect。

---

## 模型 YAML 配置

在 `ultralytics/cfg/models/26-RGBT/2026-05-14/` 下新建 YAML:

```yaml
# 顶层新增以下字段:
nc: 80
end2end: True   # [YOLO26] 端到端NMS-free
reg_max: 1      # [YOLO26] 无DFL，直接回归

# head最后一行使用 Detect (通过parse_model自动传入reg_max和end2end)
- [[16, 19, 22], 1, Detect, [nc]]
```

这样 `parse_model` 会将 `reg_max=1, end2end=True` 连同 channels 传给 `Detect(nc, reg_max=1, end2end=True, ch=[...])`。

---

## 文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `ultralytics/engine/trainer.py` | 修改 `build_optimizer()` | 集成 MuSGD (旧代码注释) |
| `ultralytics/utils/loss.py` | 新增 `E2ELoss` 类 | 注释保留旧 `E2EDetectLoss` |
| `ultralytics/nn/tasks.py` | 5处改动 | 导入/parse_model/init_criterion/loss |
| `ultralytics/optim/` | 无需改动 | MuSGD/Muon 已存在 |
| `ultralytics/nn/modules/head.py` | 无需改动 | 已支持 reg_max/end2end |
| `ultralytics/cfg/models/26-RGBT/2026-05-14/` | 新建 YAML | YOLO26风格双模态配置 |
