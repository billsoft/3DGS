# 曲面补洞、碎片清理和开源实现对照

2026-11-12。当前是独立实验，不替换 `mesh-photo-six-v4`。`mesh-photo-six-v7` 通过拓扑和运行检查，但**未通过整场景视觉验收**：球体闭合了，周围仍有错层；桌边、灯、窗帘仍有明显破损。不能把封闭网格等同于正确重建。

## 已验证的原因与未验证的假设

- 保持 v4 几何与照片来源不变，用未羽化原片替换贴图。18 个原机位方向的平均逐视图 MAE 约为 0.057/255，最大逐视图 MAE 约为 0.100/255；没有像素的最大通道差超过 20/255。灯、窗帘的碎片仍在。因此当前窄带羽化不是这些大块破损的主要原因。
- 自动球面候选得到五个机位较一致的预测深度支持，另一个机位明显冲突。这里的深度来自生成模型，不是扫描仪测量；不能直接平均成真实几何。
- 原贴图阶段删除没有合格照片来源的三角面，会把已经补好的曲面再次切开。现在区分“可靠几何但纹理未知”和“几何不可信”。前者保留并标为中性灰，不能用穿过遮挡物的颜色伪造纹理。
- 原面中心可见性不能保证整面可见，跨物体边界的大面可能采到前景/背景颜色。新增四点足迹评分只改善来源选择，是近似检查，不等于逐像素遮挡或修正几何。
- 当前深度融合仍有跨深度边界插值、冲突深度加权以及全局体素滤波。这些代码机制可能破坏薄结构，但本轮未分别量化其对每个物体的贡献。
- 桌边锯齿多数连接主表面，不属于孤立连通块。浮片清理不能解决它；当前平滑又保护开放边界，因而也保留了错误的锯齿边界。

## 本次实现

`scene_fusion/sphere_repair.py` 自动搜索局部球面，要求空间覆盖、留出点拟合、多机位深度和照片边缘证据。没有物体名称、手绘 ROI、固定房间坐标或固定相机数量。至少三个机位支持才允许替换；照片边缘接近并不是严格的物体轮廓验证。几何门槛通过后生成共享顶点的封闭球面，不允许后续平滑缩小该曲面。

本房间发现一个候选，半径约 0.2685 个归一化场景单位；5/6 个机位通过支持门槛。留出点距离中位数约 0.00583、P90 约 0.01317。生成 3968 个面，贴图导出后全部保留，开边和非流形边均为 0。其中 844 个面没有可靠纹理，使用明确的灰色材质。**周围旧表面和支架仍然破碎，部分机位照片投影也错位，所以不能宣称地球仪整体修复成功。** 检查原片与全部六个机位的放大渲染后保留此结论。

`scene_fusion/mesh_cleanup.py` 先按顶点连接关系找孤岛，再检查相对采样间距的面积、直径、与大表面的距离以及多机位深度证据。只有至少两个机位把大部分候选判为自由空间、且缺少两个可靠支持机位时才删除。未知区域、受保护的补洞曲面、长细结构及连接主表面的结构保留。当前六图实验共 306 个连通分量，仅删除 9 个小分量、56 个面；这是保守的去浮片，绝不是桌边修复或完整去噪。

这些模块接受任意数量的输入视图，但实景只验证了这一个六机位房间。合成测试覆盖平面/圆柱拒绝、旋转缩放变换、缺失球冠、自由空间冲突、细长/连接结构保护；不代表所有场景质量均通用。

## 实际查看的开源代码及适配判断

| 项目与源码 | 具体机制 | 与当前代码的差距 / 用法 |
| --- | --- | --- |
| [OpenMVS SceneRefine.cpp](https://github.com/cdcseacave/openMVS/blob/14ad2571aac650bfabf54296d31aabed7a19fdde/libs/MVS/SceneRefine.cpp) | `ImageMeshWarp`、`ComputeWindowStats`、`ComputePhotometricGradient`、`ScoreMesh`：多视图图像变形后的 ZNCC 及网格正则化共同驱动顶点优化 | 应作为下一轮几何优化对照；我们目前没有运行或移植它。仅后处理平滑不具备这些图像约束。它也不能保证补好错误拓扑。 |
| [OpenMVS SceneTexture.cpp](https://github.com/cdcseacave/openMVS/blob/14ad2571aac650bfabf54296d31aabed7a19fdde/libs/MVS/SceneTexture.cpp) | `ListCameraFaces` 统计投影面积和真正可见像素；部分遮挡降低来源质量；`FaceOutlierDetection` 排除颜色异常来源 | 我们的四点检查仍比逐像素覆盖粗糙，不能声称等价。先完善遮挡和来源评分，再考虑接缝颜色。 |
| [MVS-Texturing calculate_data_costs.cpp](https://github.com/nmoehrle/mvs-texturing/blob/f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1/libs/tex/calculate_data_costs.cpp) | 三角面顶点到相机的 BVH 射线检查、光度异常检测 | 当前不能仅用几何中心可见就允许整面附色。几何本身错位时，即使射线检查正确仍会采错图。 |
| [MVS-Texturing view_selection.cpp](https://github.com/nmoehrle/mvs-texturing/blob/f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1/libs/tex/view_selection.cpp) | mapMAP 全局标签优化；数据项和邻接项共同决定照片所属区域 | 当前局部标签迭代不是 mapMAP；下一轮可与该工具输出对照，不能用势力范围硬裁代替可见性。 |
| [PGSR train.py](https://github.com/zju3dv/PGSR/blob/main/train.py) | 单视图深度法线约束；邻近机位深度往返投影误差；遮挡感知权重；局部图像块 NCC | 适用于重新优化公共几何，而非直接拼接独立 PLY。代码已阅读，未运行其训练或移植损失函数。 |
| [2DGS train.py](https://github.com/hbb1/2d-gaussian-splatting/blob/main/train.py) | 深度畸变和表面法线一致性正则化，联合图像重建损失训练 | 减少沿射线分散的几何并约束表面；不是可以直接套在已有 mesh 上的滤波器。尚未运行或移植。 |
| [Open3D 连通块示例](https://github.com/isl-org/Open3D/blob/main/examples/python/geometry/triangle_mesh_connected_components.py) | 按连通块统计面数并删除小块 | 本实现使用 SciPy 计算连接关系，并额外要求自由空间反证，避免只按大小删除真实细节。 |

以上引用用于代码与方法对照，没有把第三方源码复制进项目，也没有将尚未运行的工具报告为已验证改进。OpenMVS 官方最新 release 元数据本次未提供可直接下载的二进制附件；本轮没有构建其原生 Windows 工具链。

## 下一轮应做的受控实验

1. 固定相机标定和相同输入，导出全景的标定透视视图。来自同一全景的多个方向共享一个光心，不能当作新增独立机位或虚构视差。
2. 基线网格不变，先用成熟贴图工具做对照，分离“照片来源/遮挡错”与“网格轮廓错”。统计逐机位局部误差，而非只看全场均值。
3. 再对比 OpenMVS 的图像约束网格优化，或 PGSR/2DGS 联合重建。独立生成 PLY 只能作为带不确定性的初始化；原照片才是观测约束。
4. 对桌边、细杆、灯的测试必须包括轮廓误差、孔洞面积、自由空间冲突和相邻视点稳定性。通用检测选区域，不硬编码物体位置。不以颜色更顺眼代替几何正确。
5. 平滑限制在同一可信表面内；跨遮挡边缘、真实棱角和细结构禁止平均。没有观测的背面只能保守补几何，不能声称恢复了真实细节。

## 复现实验

```powershell
python scene_fusion/sphere_repair.py --source four-room/scenes/geometry-quads-v1 --mesh four-room/scenes/mesh-geometry-six-v4/surface.npz --warped --out four-room/scenes/mesh-geometry-six-v5
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-quads-v1 --geometry four-room/scenes/mesh-geometry-six-v5/surface.npz --warped --footprint --clean-fragments --out four-room/scenes/mesh-photo-six-v7
python tests/test_sphere_repair.py
python tests/test_mesh_cleanup.py
python tests/test_textured_mesh.py
node tests/check_photo_mesh.mjs mesh-photo-six-v7
```

输入是私有测试数据，不进入公开仓库。预览 `mesh-photo-room.html?scene=mesh-photo-six-v7&surface=0`；默认仍为 v4。球面按钮仅把原始机位镜头转向候选并缩小 FOV，不伪造新拍摄位置。点击机位按钮恢复普通 FOV。

18 个单元测试通过；浏览器捕获 18 个原机位方向、6 个中点、几何/所属区域、球面几何和六个机位近景，无运行错误。导出 OBJ/MTL/贴图 ZIP 可解压，面数 688703。上述运行与拓扑通过不改变视觉未达标的结论。
