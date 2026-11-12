# 把原始机位漫游蒸馏成单一 PLY

## 目标和结论

用户希望把“更大 · 过渡更短”的 44% 原始漫游效果，提前采样并训练到一份普通 3DGS PLY，避免依赖多模型切换播放器。本次在当前 Windows 的 RTX 4090 上实现并生成 `tour-distilled-six-v1`，不使用 WSL 或远程训练。

文件包含 917420 个高斯、二阶球谐颜色、位置、旋转、尺度和透明度。它是单一静态模型，不包含原始六份 PLY、相机选择器、照片覆盖层或外观 MLP。它需要支持高斯 PLY 和二阶 SH 的渲染器；把它当普通 XYZ 点云或只读取 DC 颜色会损失效果。

**这是有损蒸馏实验，尚未达到所有原始机位不退化的要求。** 不替换原始漫游或已接受的融合默认入口。灯具和墙面相对未训练的静态初始化改善明显，但白板文字、路由器/细杆、部分桌面及过渡区域仍可能模糊、错位或缺损。不能用平均分掩盖这些问题。

## 为什么不能直接“扫一遍把点存起来”

播放器目标为 `T(camera) = sum_i w_i(camera) * render(native_i, camera)`。权重属于观察相机的位置，不属于高斯点的三维位置；把它乘到点透明度上再合并，通常不等于分别渲染后的图像混合，因为遮挡与透射率的组合是非线性的。

原播放器还包含停止平移后转为单一来源、加载完成后的时间平滑和交界迟滞。同一位置可以因运动历史显示不同画面。静态 PLY 不存这些状态。因此本次明确固定为无历史、无“停下清晰优先”的确定性 44% 空间规则。在每个原机位仍为对应原始模型 100% 权重。

50% 会使中间渐变带消失，变为最近机位的硬切换，而非更多重叠。这个突变目标更难被一个通常连续变化的静态模型逼近。本轮采用 44%，不宣称同时复现移动和停止两种模式。

虚拟相机可以增加监督视角，但不能增加实拍信息。独立生成模型之间的错误背面、错误深度或互相矛盾的轮廓仍会进入目标画面；扫描合成视图不能证明这些几何真实。

## 已检查的参考实现

- [Fed3DGS build_global_model.py](https://github.com/DensoITLab/Fed3DGS/blob/main/gaussian-splatting/build_global_model.py)：`distillation` 先渲染局部/全局模型目标，再优化全局表示。代码还包含外观 MLP、位置编码和逐视图外观向量，因此不能直接把该完整表示当作普通 PLY 的解决方案。
- [Distilled-3DGS train.py](https://github.com/lt-xiang/Distilled-3DGS/blob/master/train.py)：检查了 teacher/student、多教师渲染监督与几何分布相关代码。本实验没有复制该项目的训练器，也没有实现其全部损失或剪枝策略。
- 可微渲染沿用项目现有的 [gsplat](https://github.com/nerfstudio-project/gsplat)。使用自编采样、普通 SH2 学生模型和现有已做梯度检查的图像损失。

## 实际流程

1. 从 `native.json` 读取任意数量的原始模型、相似变换、拍摄位置和邻接路线。校验原始 PLY 哈希，原始文件保持不变。
2. Python 44% 权重与实际 `native-tour-policy.js` 对照测试，包括偏离路线的位置、等距点和原机位。相似变换同时作用于高斯中心、尺度和旋转。
3. 每个位置采六个方向。六图数据共 108 个虚拟位置、648 个方向；其中 72 个位置 / 432 个方向训练，36 个不同位置 / 216 个方向留出。训练包括六个原机位、附近横向偏移和路线的 12/25/38/44/50/56/62/75/88%；留出路线位置为 19/47/53/81%，另有不同距离的附近偏移。
4. 每个目标都渲染完整原始 PLY，再做图像级加权。初始静态模型按点到拍摄中心的最近距离选取来源，保留 917420 点；这仅是优化起点，不被称为播放器的等价空间分割。
5. 优化位置、旋转、尺度、透明度与二阶 SH。用 RGB、局部 SSIM、梯度和覆盖损失监督，原机位额外采样。纯来源画面使用很弱的合成深度约束；混合两张画面的深度不作为真实表面。位置相对初始点限移，尺度有上下界，SH 有正则化。未实现自适应增点或真实多视图三角化。
6. 第一阶段 384 像素、1800 步；第二阶段 768 像素、2400 步，教师在 GPU 即时渲染高分辨率目标。没有把 384 图简单放大当作高分辨率监督。
7. 在 384/768 重新渲染原机位和未训练位置，导出标准 SH PLY 并逐字段回读。浏览器只加载该文件，继续做同相机渲染对照。

## 数值结果

参考是完整原始 PLY 的 44% 合成漫游画面，不是新实拍。表中的“初始”是本实验未训练的静态分区模型，不是此前已接受融合版。

| 分辨率与视图 | 方向数 | 初始 SSIM → 学生 SSIM | 初始 MAE → 学生 MAE |
| --- | ---: | ---: | ---: |
| 384 原机位 | 36 | 0.89702 → 0.96948 | 0.02232 → 0.00962 |
| 384 未训练位置 | 216 | 0.89056 → 0.94943 | 0.02449 → 0.01333 |
| 768 原机位 | 36 | 0.91503 → 0.97493 | 0.02159 → 0.00872 |
| 768 未训练位置 | 216 | 0.91031 → 0.95750 | 0.02381 → 0.01260 |

颜色误差范围为 0–1。数值门槛要求每个原机位方向 SSIM ≥0.97、MAE ≤0.01，且教师高覆盖像素不出现学生 alpha <0.5。768 下有 17/36 个方向未全部满足，因此整体保真门槛为失败。最差方向为 `source-2-2`，SSIM 约 0.89155；白板和路由器的目视退化也明显，不能略过。

768 下原机位合计 125 个高覆盖参考像素变为低 alpha；未训练位置合计 8405 个。这是像素计数，不是物理孔洞数量，也不意味着采样位置以外均无洞。浏览器与可微渲染的误差单独记录在私有实验目录的 `browser/parity.json`。

PLY 大小 139448792 字节，二阶 SH 回读完全一致。文件脱离训练数据和原始 PLY 后仍可加载；相机按钮、验证报告和路线仅供预览，不是渲染该文件的必要条件。

实际浏览器检查了 7 个原机位/中间方向，网络日志只请求一份 `final/fused.ply`，无运行错误。按浏览器实际相机矩阵重渲染后，网页与 GPU 学生模型平均 RGB 差为 0.00377（约 0.96/255），因此本次离线与网页结果基本对应；这不等于它与原始模型无差异。公开摘要见 [experiment summary](../experiments/2026-11-12-tour-distillation/summary.json)。

## 复现

```powershell
./run-refit-gpu.ps1 scene_fusion/tour_distill.py prepare --native four-room/scenes/native-source-six-v1 --out four-room/scenes/tour-distilled-six-v1 --size 384
./run-refit-gpu.ps1 scene_fusion/tour_distill.py train --out four-room/scenes/tour-distilled-six-v1 --steps 1800
# 先保留第一阶段检查点，再继续高分辨率训练。
Copy-Item four-room/scenes/tour-distilled-six-v1/optimized.npz four-room/scenes/tour-distilled-six-v1/phase1.npz
./run-refit-gpu.ps1 scene_fusion/tour_distill.py train --out four-room/scenes/tour-distilled-six-v1 --resume four-room/scenes/tour-distilled-six-v1/phase1.npz --train-size 768 --lr-scale 0.4 --steps 2400
./run-refit-gpu.ps1 scene_fusion/tour_distill.py evaluate --out four-room/scenes/tour-distilled-six-v1 --eval-size 768 --evaluation-folder evaluation-hd
python scene_fusion/tour_distill.py export --native four-room/scenes/native-source-six-v1 --out four-room/scenes/tour-distilled-six-v1 --evaluation-folder evaluation-hd
python tests/test_tour_distill.py
node tests/check_tour_distill_browser.mjs tour-distilled-six-v1
./run-refit-gpu.ps1 scene_fusion/tour_browser_audit.py --out four-room/scenes/tour-distilled-six-v1
```

数据路径只示范当前私有实验，不进入代码仓库。没有固定六机位循环；不同场景必须提供其自身已验证的相机元数据。预览：`scene-room.html?scene=tour-distilled-six-v1&inspect=3`；同视点对照：`tour-distill-report.html?scene=tour-distilled-six-v1`。
