# 曲面保边、照片分区与低频重建试验

本轮保留用户已接受的 `baseline-mesh-guide-20261113`。新增结果是独立
取舍版本，不自动替换该基线，也不修改上游单一高斯 PLY 和原始漫游。

后续用户已认可自适应和平滑优先两版作为可用预设。它们已接入
[通用完整流程](reconstruction-workflow.md)，原试验的缺陷与取舍记录仍保留，
不将用户认可解释为所有区域全面优于旧版。

## 定位出的不同问题

1. 在同一浏览器相机分别查看原片贴图、未贴图面标记、法线和照片来源。
   棚顶的大部分浅色碎片与小块照片来源切换相对应；少量灰面确实没有
   可用贴图。不能把所有色块叫作几何孔洞。
2. 基线曲面在 114 个已检查方向没有射线漏空，但局部法线起伏仍很大。
   射线命中也可能命中错误的后层，零漏空不代表几何精确。
3. 新低频重建结果转为 float32 后有重合顶点与零面积连接三角形。直接
   删除退化面会留下拓扑缝。先按中位边长的 0.001 倍焊接数值尺度顶点，
   再去掉退化面，最终自适应 / 中等低频网格均恢复零开放边。这不修复大洞，
   也不保证无自交或可制造实体。

## 实际实现及对照

- `mesh_feature_denoise.py`：沿拓扑邻域双边滤波面法线，再用切平面约束
  更新顶点。冻结边界 / 非流形顶点，限制总位移，发生翻面或严重塌缩时
  回退相邻顶点。主要平面法线 RMS 改善约 4%–13%，但棚顶外观改善有限。
- `texture_patch_consensus.py`：只在原本可见、颜色相容的候选照片中整理
  小连通分区。不能跨遮挡凭空取图；可选的多照片颜色离群拒绝会误伤
  灯等薄结构，在交付的自适应 / 中等低频版本中没有启用。
- `texture_graphcut.py`：可选 PyMaxflow alpha-expansion，在网格面图上优化
  Potts 照片分区。接缝边数从 32,713 降到 15,796，但灯附近误色扩大，
  2 号机位棚顶原照片 MAE 从 0.01434 升至 0.01502，因此不推荐为默认。
  通过穷举小图验证二标签全局最小能量、单次扩展最优解和可见性限制。
- `planar_guide_refine.py`：使用既有自动估计、至少三个机位深度支持的
  平面，在局部 PCA 平坦且法线相容的引导点上有限投影。提高 Poisson
  分辨率后仍未有效消除色块，不选作本轮推荐对照。
- `coarse_surface.py`：体素聚合带法线引导点、较低分辨率 Poisson 重建、
  Taubin 平滑。不是将坐标缩小再放大；改变的是采样与曲面的空间频率。
  所有长度相对场景范围或采样间距设置，没有物体名字和固定机位数量。
- 自适应模式：在至少两张原照片有明显梯度且射线可见的位置保留较密
  引导点，其余区域降低采样密度。照片边缘仅用于保留采样，未强行变成
  几何折线。保留 104,883 个边缘支持样本，总引导样本 208,968 个。
- `mesh_numeric_weld.py`：显式数值焊接并记录开放边、非流形边和微小面。
  自适应候选 702,412 面，0 开放边、28 非流形边；中等低频候选 338,390
  面，0 开放边、24 非流形边。数量减少不构成几何真实性证明。

最强的低频候选约 8.6 万面，棚顶平顺，但灯、支架等薄物体明显损坏；
保留用于研究，不作为推荐结果。中等低频与自适应也仍有灯 / 电视边缘
弯曲和反射接缝，不能称为全面超过基线。

自动矩形轮廓检测得到 768 个二维候选及 98 个双机位配对，但没有通过
当前严格第三机位验证的矩形。原因可能包含轮廓匹配和标定残差，不能
仅据此归因于其中一项。本轮没有放宽验收强行插入电视或门的矩形。
低频滤波能减少锯齿，不会自动获得准确直线、直角或标准球体。

## 可重现命令

Open3D 命令使用独立 `requirements-mesh.txt` 环境；贴图使用原 CPU 环境。
PyMaxflow 仅在 `--graphcut` 时需要，独立可选依赖见
`requirements-mesh-graphcut.txt`，没有捆绑第三方源代码。

```powershell
python scene_fusion/coarse_surface.py --guide four-room/scenes/geometry-smooth-guide-poisson-six-v1/geometry-guide.ply --cameras four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/mesh-adaptive-geometry-six-v1 --relative-voxel .004 --depth 9 --iterations 3 --edge-source four-room/scenes/geometry-smooth-depth-strong-six-v1 --reference-mesh four-room/scenes/geometry-smooth-guide-poisson-six-v1/surface.npz
python scene_fusion/mesh_numeric_weld.py --mesh four-room/scenes/mesh-adaptive-geometry-six-v1/surface.npz --cameras four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/mesh-adaptive-geometry-six-v2
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-smooth-depth-strong-six-v1 --geometry four-room/scenes/mesh-adaptive-geometry-six-v2/surface.npz --out four-room/scenes/mesh-adaptive-photo-six-v2 --warped --footprint --texture-width 7680 --smoothing 0 --visibility-cache four-room/scenes/mesh-adaptive-geometry-six-v2/visibility.npz --coherent-patches
```

中等低频使用 `--relative-voxel .004 --depth 8 --iterations 12`，不传
`--edge-source` / `--reference-mesh`；之后同样进行数值焊接及原片贴图。
强版本参数为 `.007 / 7 / 12`。引导 PLY 是普通表面点云，不是显示用高斯。

## 验证与参考

最终数值和模型哈希见 `experiments/2026-11-13-mesh-lowpass/`。原基线的
14 个产物哈希复核一致。两个候选分别完成 18 个原机位视向、6 个中点及
96 个斜视浏览器渲染检查；各 114 个射线检查方向没有漏空。浏览器通过
指资源正常加载和无运行异常，不能解读为所有画面质量合格。

最终 2 号机位棚顶的原照片归一化 RGB MAE：基线 0.014341，自适应
0.015059，中等低频 0.015909。三处主要平面的法线 RMS 分别改善约
3%–20%，但拟合支持面积也略有变化。这些结果说明表面更平顺，却没有
证明照片保真更高；局部灯具退化仍存在，两个结果均保持独立候选状态。

同视角比较入口为 `/mesh-smooth-compare.html`。页面的四组对照截图及
六个预览 / 下载链接已验证可加载。

`capture_mesh_diagnostics.mjs` 保存实际浏览器相机矩阵及四种显示模式；
`mesh_photo_compare.py` 按该矩阵重投影原照片，比较原机位外观，不用近似
截图角度代替。仍须结合细结构局部图和斜视检查，不能只看全图均值。

- [Bilateral Normal Filtering for Mesh Denoising](https://doi.org/10.1109/TVCG.2010.264)：
  法线滤波再更新位置的经典方法；本项目为受启发的独立、有位移限制实现。
- [Guided Mesh Normal Filtering 开源实现](https://github.com/bldeng/GuidedDenoising)：
  查阅其项目和算法说明，没有集成其 Qt 程序或复制 LGPL 源码。
- [Open3D 网格滤波](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html)：
  本轮实际调用 Open3D 的 Taubin 平滑、Poisson、体素降采样及顶点焊接。
- [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing)：查阅
  `libs/tex/view_selection.cpp` 和 `texture_view.cpp` 的分区 / 数据项流程。
  本轮没有使用其 mapMAP 求解器，独立实现可见性约束的 Potts alpha-expansion。
- [CGAL Polygon Mesh Processing](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html)：
  查阅修复及保特征重网格接口，本轮没有编译或调用 CGAL。

照片、Mesh 和贴图仍保留在本机，不提交公共仓库；代码、参数和数值验证留档。
