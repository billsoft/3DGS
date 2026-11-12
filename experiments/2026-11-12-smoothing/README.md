# 毛刺平滑消融：未发布新模型

用户请求减轻 `shape-free` 的毛刺，重点检查地球仪。四类候选都没有达到“明显去刺且保留画质”的标准，当前 `/shape-room.html` 与 PLY 未改变。`smoothing-review.html` 仅展示本轮对照。

- `smooth_gaussian_surface.py`：同法线、双向切向支撑、平衡邻域的加权中位点到面平滑。中档最大移动 0.0014 模型单位；26 个来源方向全部指标下降，近景去刺不明显。
- `smooth_footprints.py`：局部朝向/长度异常约束。调整 231,557 个朝向、12 个长度；地球仪近景目视无明确收益，未进入完整来源评分或发布。
- `smooth_normal_motion.py`：从历史初始化提取法线位移，在不跨层的邻域图扩散，保留切向位移。中档随后固定几何，用 `refit_train.py --mode color` 拟合 500 步。来源平均 SSIM 下降，新增低透明度像素 412，拒绝。
- `repair_normal_outliers.py`：邻域法线投票识别朝向异常，仅转动 6,901 个高斯，最大 25°，中心/颜色/不透明度/尺寸严格不变。新增低透明度像素 0，但平均 SSIM 略降，地球仪尖刺仍明显，拒绝。

所有输入是同一份私有 `four-room/shape-free/optimized.npy`。位移方法的参考为 `four-room/refit-data/initial.npy`；该历史初始化含此前物体几何先验，不能宣称这是无需先验的纯照片方法。此次没有增加球体拟合、灯具补片或语义专用编辑。`inspect_smoothing.py` 的地球仪坐标仅指定诊断相机，不参与修改算法。

三个完整量化候选使用 `refit_evaluate.py --data four-room/shape-data --baseline four-room/shape-free/optimized.npy --candidate CANDIDATE --out OUTPUT`；指标使用当前 shape 模型的高覆盖掩码，因此基线分数与上一轮使用另一掩码的分数略不同。26 个来源方向来自同四张照片，并非独立留出数据；12 个中点没有真值。alpha 阈值与验证局限见报告页面。

结论仅限这些参数和本房间：后处理削弱几何突起会改变像素覆盖和已有颜色拟合，轻微修改无明显去刺效果，较强修改可退化。未证明根因已全部定位，也未证明所有平滑方法都无效。后续方向是联合几何正则化和多视图重投影/遮挡约束，而不是继续增大全局平滑强度。

复现候选的参数见脚本 argparse；在原生 Windows CPU 环境运行提案生成器，在已有 GPU 环境运行渲染与颜色拟合。私有照片、数组、模型和截图不提交；此目录仅保存聚合数值和失败结论，避免重复发布退化方法。
