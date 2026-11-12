# 固定配准的高斯形状对照

从已接受的 `2026-11-12-refit-surface-guard` 私有模型开始，两组均固定相机和全部高斯中心，点数均为 3,166,125。同一初值、照片、随机种子和 1200 步；对照只训练颜色/尺寸，候选额外训练四元数和不透明度。默认训练入口行为不变，需显式 `--shape-free` 启用。

| 1024 来源视向指标 | 基线 | 继续训练对照 | 形状候选 |
| --- | ---: | ---: | ---: |
| 平均 SSIM | 0.918674 | 0.924319 | 0.929793 |
| 平均绝对颜色误差 | 0.019491 | 0.018943 | 0.018223 |
| 26 方向中 SSIM 改善数 | — | 25 | 26 |
| 38 方向新增低透明度像素 | — | 0 | 0 |

512 复核：平均 SSIM 0.901257 → 0.907575，25/26 来源方向改善，最差方向下降 0.000437；38 方向新增低透明度像素为 0。每档使用各自基线 alpha > 0.95 的固定掩码，候选 alpha < 0.5 计作新增低透明度像素。这不涵盖已有空洞、错误颜色或所有视点的遮挡错误。

结论：固定配准和中心仍有额外收益，说明被冻结的高斯形状/透明度也是限制因素，不能把差异全部归因于配准。此试验未分别隔离朝向和透明度，也未排除剩余相机误差。平均绝对透明度变化仅 0.000285，旋转变化中位数 2.83°，不能据此直接断言哪项单独贡献最大。

人工检查原机位灯、窗框/白板与中间视点灯、桌面：局部轮廓、文字略有改善；灯体厚度和桌边毛糙依然明显，中点改善较小。没有使用灯具专用处理或照片覆盖。该版是增量候选，不替换已接受基线；不是完整修复，更不是新机位重建精度的证明。26 个来源方向均来自同四个拍摄位置，照片参与拟合；12 个中点没有独立真值。

## 复现

使用仓库原生 Windows GPU 环境，沿用既有私有 `refit-data-hd` 的 `targets.npy`、`dataset.json`、`neighbors.npz`，复制至 `four-room/shape-data`。将已接受 `refit-surface-guard/optimized.npy` 作为该目录的 `initial.npy`。邻接图沿用此前的对应索引与权重，没有重建相机或邻接图。

```powershell
./run-refit-gpu.ps1 -PythonArguments @('fusion_next/refit_train.py','--data','four-room/shape-data','--out','four-room/shape-free','--steps','1200','--warmup','0','--graph-weight','1.5','--edge-weight','0.3','--color-lr','0.0008','--position-lr','0','--scale-lr','0.0004','--shape-free')
```

对照去掉 `--shape-free`，输出改为 `shape-control`。每组用 `refit_evaluate.py --data four-room/shape-data --candidate .../optimized.npy --out .../eval` 检查；低分辨率传入 `refit-data` 并显式指定接受基线 `--baseline`。尺寸在初值 0.8–1.25 倍内，四元数每步归一，不透明度 logit 限制在 [-2,8]。本次初值全部位于该范围，不存在初次截断引起的突变。两组使用相同保覆盖与中间视点基线信任项。

`check_shape.py` 检验中心严格不变、有限值、尺寸范围和旋转归一；`export_refit.py --initial four-room/shape-data/initial.npy --allow-opacity-change` 才允许导出本候选。PLY 往返读取必须逐值一致。导出仍需既有 `GS_INPUT_DIR`、`GS_OUTPUT_DIR` 配置和相机标定缓存。

`publish_shape.py` 将新入口发布为 `shape-room.html` 和 `shape-report.html`，`?version=baseline` 指向已接受快照。已有 `refit-room.html` 不变。任务专用浏览器 CDP 中设置 `PREVIEW_URL=http://127.0.0.1:7870/shape-room.html` 后运行 `node tests/check_refit_browser.mjs four-room/shape-free`，核验单模型加载并读回实际 WebGL 画布。

本目录只包含代码说明和聚合验证数据；原片、模型、截图不公开。
