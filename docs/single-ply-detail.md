# 49% 原始漫游的单一 PLY 细节试验

保留旧多合一算法、`tour-distilled-six-v3`、`tour-appearance-six-v1` 和原始
六模型漫游。新试验使用独立目录；以前的默认训练参数不变。旧 PLY 哈希见
`experiments/2026-11-13-single-ply-detail/previous-models.json`。

## 本轮方法

教师继续完整渲染原始六份 PLY，并按 49% 固定空间规则混合图像。
不使用已撤回的原片附色，不把多份 PLY 的透明度直接相乘后拼接。
从已保留的单一模型初始化，训练位置、尺度、旋转、透明度和 SH2 颜色。

本轮的可选改动：

- 对屏幕投影梯度大且覆盖范围大的高斯进行两轮细分，每轮最多 30 万个父点，
  增加复杂轮廓附近的表达能力。细分是优化初始化，不宣称保持渲染严格不变。
- 每帧除原有 RGB、结构、梯度和覆盖损失外，再计算 32×32 局部 RGB 与色度
  误差，对最差四分之一图块增加训练权重，不指定灯、桌子等物体。
- 使用较小学习率并逐渐衰减，避免刚开始续训就大幅改变已接受外观。
- 保留原有宽视角及路线监督，加入自动生成的 50° 机位/近机位视角；60°
  留出检查。机位数量来自场景数据，不固定为六个。

```powershell
.\run-refit-gpu.ps1 scene_fusion/tour_refine.py `
  --reference four-room/scenes/tour-appearance-six-v1 `
  --out four-room/scenes/tour-detail-single-six-v1 `
  --views four-room/scenes/native-teacher-audit-v1/detail-views.json `
  --sizes 640 1024 1536 --steps 3200 --core 0.49 `
  --lr-scale 0.35 --final-lr-ratio 0.1 --patch-weight 0.15 `
  --max-points 2200000 --split-count 300000 --split-steps 500 1300
.\run-refit-gpu.ps1 scene_fusion/tour_distill.py evaluate `
  --out four-room/scenes/tour-detail-single-six-v1 `
  --eval-size 640 --evaluation-folder evaluation-640 --local-checks
```

`--patch-weight 0`、`--lr-scale 1`、`--final-lr-ratio 1` 和原有细分步数为
默认值，保持旧训练流程。新参数不改变已保存的旧模型。

## 验证方式

`--local-checks` 增加色度误差、局部图块 95 分位和最大误差，并保存最差
退化方向列表。与同相机的原始漫游和前一份单一 PLY 比较，不用全图均值
替代局部验收。掩码只用于有高覆盖的参考区域，不能证明未知区域正确。

浏览器检查可增加 `--detail`：除 90° 的机位及路线视点，还捕获 50° 的
水平、抬头、低头画面，并确认网络只请求一个 `final/fused.ply`。
PLY 导出必须通过字段回读检查。几何和 SH 都存储在文件内；播放时不依赖
六份原始 PLY、训练图像或机位选择网络。

这仍是有损蒸馏。确定性 49% 空间规则不等同于播放器的停下清晰优先及时间
过渡状态；增加细节也可能放大教师间的几何冲突，需要逐视点检查。

## 第二轮：交接附近补采样

首轮 408 个检查方向平均改善，但个别交接方向退化。原取样在 44%、50%、
56% 附近较稀，第二轮按场景邻接图自动加入 48%、49%、51%、52% 位置的
训练视角，并在 47.5%、48.7%、51.3%、52.5% 留出检查。它们不与原有
留出位置重合。`tour_detail_views.py --transitions` 生成这些方向，适用于
任意机位数量；两轮都没有手工圈定灯、桌子或电视区域。

```powershell
python scene_fusion/tour_detail_views.py `
  --dataset four-room/scenes/tour-appearance-six-v1/data/dataset.json `
  --out four-room/scenes/native-teacher-audit-v1/detail-transition-views.json --transitions
.\run-refit-gpu.ps1 scene_fusion/tour_refine.py `
  --reference four-room/scenes/tour-appearance-six-v1 `
  --resume four-room/scenes/tour-detail-single-six-v1/optimized.npz `
  --out four-room/scenes/tour-detail-single-six-v2 `
  --views four-room/scenes/native-teacher-audit-v1/detail-transition-views.json `
  --sizes 640 1024 1536 --steps 1800 --core 0.49 `
  --lr-scale 0.18 --final-lr-ratio 0.2 --patch-weight 0.15 --no-split
```

同一批 408 个检查方向：首轮/第二轮 MAE 为 0.0107653/0.0106728，
SSIM 为 0.960958/0.961326，局部图块 P95 为 0.0393861/0.0390915。
第二轮小幅改善，但 1–6 交接处电视的局部退化未消除，不能称作全部优于旧版。
新文件只能作为独立对照候选，旧算法及旧模型继续保留。

`tour_teacher_conflicts.py` 可在一个指定检查视角渲染最近两份原始模型，
比较其同相机 RGB 和合成深度。合成深度不是测量真值；分歧不能直接证明
其中哪份模型正确，但可以定位静态模型需要折中拟合的来源冲突。

## 高清检查结果与导出

第二轮在 1024×1024 的 552 个方向复核：36 个原机位方向 MAE 从
0.0074173 降至 0.0044127，SSIM 从 0.982455 升至 0.984865；516 个
未训练视点或方向 MAE 从 0.0151826 降至 0.0135963。这里的参考是原始
漫游渲染，不是真实照片；误差下降不能表述为相同比例的清晰度提升。

所有检查方向的局部色度误差从 0.0033264 降至 0.0024310，图块 P95
均值从 0.0516053 降至 0.0497822。但 1–6 交接电视方向的 P95 从
0.217365 升至 0.230589。严格原机位保真门槛仍未通过，原有斑点和部分
轮廓模糊也仍存在。因此保留旧默认入口，仅发布独立候选供下载对照。

同相机的来源冲突检查中，两份原始模型的合成深度相对差中位数约 5.1%，
约 50.2% 的有效像素差异超过 5%。电视轮廓位置也不同，说明监督图像本身
存在几何冲突；这不是两模型配准误差大小的测量，也不证明所有缺陷都来自配准。

```powershell
.\run-refit-gpu.ps1 scene_fusion/tour_distill.py evaluate `
  --out four-room/scenes/tour-detail-single-six-v2 `
  --eval-size 1024 --evaluation-folder evaluation-1024 --local-checks
python scene_fusion/tour_distill.py export `
  --native four-room/scenes/native-source-six-v1 `
  --out four-room/scenes/tour-detail-single-six-v2 --evaluation-folder evaluation-1024
node tests/check_tour_distill_browser.mjs tour-detail-single-six-v2 --detail --hd
```

输出 `final/fused.ply`：1,967,420 个 SH2 高斯，299,048,793 字节，导出字段
回读精确一致。独立播放器只需要该文件；项目预览所用机位按钮不是额外模型。
回退使用旧 `tour-appearance-six-v1` 入口即可，无需重新训练。

1280×1280 浏览器的 13 个方向检查通过：只请求一份 PLY，SH2 已启用，
没有运行异常。逐相机比较浏览器与 GPU 输出的平均 MAE 约 0.004356，
两渲染器仍不完全一致。实看地面及天花板还存在斑点和软化，不能把离屏
误差改善说成屏幕效果已经无损一致。数值记录见
`experiments/2026-11-13-single-ply-detail/`。
