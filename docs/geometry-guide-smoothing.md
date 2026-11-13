# 单一高斯到 Mesh：几何磨皮与原机位优先试验

本轮把显示用的单一高斯与重建用的几何引导分开。原有多合一算法、原始
机位漫游、全部基线 PLY 和 Mesh 保留，新增参数默认关闭。

2026-11-13 用户接受 `geometry-smooth-guide-photo-six-v1` 为新的 Mesh + 贴图
基线；[验收与文件哈希](../experiments/2026-11-13-geometry-guide/accepted-baseline.json)
将这次视觉验收与具体产物绑定。以下消融记录保留，接受基线不表示细结构
和所有视角已经准确。

## 下一阶段：轮廓细节（尚未实现或验证）

当前深度每面 768 像素，TSDF resolution 512、体素约 0.0292 场景单位。
TSDF 表面从约 170 万面减到 120 万面，再均匀采样约 50 万个表面点，
经过点 / 法线平滑与 depth 9 的 Poisson 重建。这些是可能损失细节的环节，
尚未用逐项消融确认电视、灯、地球仪各自误差的主要来源。

下一轮保持本基线作对照，优先比较增加深度采样精度、减少边缘减面损失、
曲率与观测置信度驱动的局部平滑。轮廓约束必须有多视图及遮挡证据，
不能将照片中的所有颜色边缘当成真实几何边缘。平面、球面等拟合仅在
多视图支持和残差检查通过时启用，不能将特定物体的位置或形状写死。

贴图边界应检查前后景串色及斜视投影；同一连续表面内才做窄带羽化。
电视屏幕反射随视角变化，普通固定贴图无法同时保留各机位不同的反射。
该问题需要与轮廓几何分开评价，不能靠全局增大平滑或颜色混合解决。

验证除原来的地面 / 墙面闭合与斜视覆盖，还应增加细结构的同视角轮廓、
几何无贴图视图，以及多机位 / 中点局部对照；不以更高面数或零漏空单独
判定进步。上述内容是下一轮假设，不是已完成的改进。

## 实际流程

```text
已接受的单一 PLY（等价优化检查点）
  → 原机位和路线中点的合成深度 / alpha
  → 保深度边界的几何双边平滑
  → TSDF 临时表面、有限碎片清理和补面
  → 采样带法线的几何点云
  → 三维位置 / 法线联合平滑，输出 geometry-guide.ply
  → Screened Poisson 连续曲面重建
  → 可见性检查、原始 7680×3840 全景贴图
```

引导 PLY 是普通带法线点云，**不是可播放的 3DGS PLY**。没有把高斯中心
直接当成真实表面，也没有修改显示用高斯的颜色。中间 TSDF 是生成引导
样本的临时表面，最终交付的是重新生成并贴图的 Mesh。

当前提取只看深度及 alpha，不看 RGB；颜色斑点不会直接变成洞。错误的
多层深度、表面不连续以及后续删面才可能造成洞。原机位的视觉正确也不
证明其所有新视点深度正确。高清贴图能恢复外观细节，不能恢复被磨掉的
几何轮廓或纠正错误形状。

## 几何处理

`depth_geometry_smooth.py` 在对数深度上做距离相对的双边滤波。strong
配置为半径 5、范围 sigma 0.025、4 次、相对深度变化上限 3.5%。明显深度
跳变、未知区域邻域及图像边缘冻结；alpha 和照片保持原样，不填未知深度。

`point_geometry_smooth.py` 使用带法线表面样本的 24 邻域，按空间距离、
法向距离和法线方向加权，平滑法线并沿拟合局部法线更新位置。法线余弦
小于 0.8 的邻居不混合；4 次迭代的累计位移限于一个原始表面采样间距。
本次间距约 0.03010 场景单位，不能直接解释为米。原始高斯不受影响。

## 消融检查

同一 `tour-appearance-six-v1`、同一 72 份深度、768 分辨率、TSDF
resolution 512、最多 120 万面，114 个同相机检查方向（含 96 个斜视 / 路线）：

| 输入或处理 | 未命中网格的射线 |
|---|---:|
| 原深度 | 71,303 |
| 中等深度平滑 | 68,324 |
| 强深度平滑 | 63,583 |
| 强平滑但仅保留原机位深度 | 115,111 |
| 强平滑 + 中间视点原机位深度支持过滤 | 88,543 |
| 强平滑 + 自动平面深度约束 | 66,013 |
| 强平滑 + 清理补面 + 贴图 | 63,366 |
| 再增加连续曲面重建 | 0 |
| 再增加三维点 / 法线平滑（本轮交付候选） | 0 |

最后两项的补面包含推断，漏空为零不是精度证明，也不保证所有未知视点。
自动平面约束额外引入局部异常面，未选为默认候选；原机位独占和当前支持
过滤虽然减少不可靠虚拟输入，却损失覆盖，也没有作为默认修复发布。

三维点 / 法线平滑相对仅连续重建：前 3 个主要拟合平面邻域的法线 RMS
从 12.10° / 14.09° / 7.97° 降至 10.54° / 12.00° / 6.84°。这些是对估计
平面的诊断，须同时看支持面积，不能当作真实形状误差。非流形边从 299
降为 170，仍不是可用于制造的合格实体。浏览器 18 个原机位方向、6 个中点、
几何与分区模式，以及 96 个斜视检查通过；实看仍有细结构、反射和贴图缺陷。

## 原机位加权

`tour_refine.py --anchor-probability .65 --anchor-weight 2` 增加原机位抽样
和损失权重，按真实相机中心判断，包含 50° 原机位方向，近机位不冒充原
机位。默认不改变旧采样策略。一次固定几何外观续训中，1600 步实际抽到
原机位 1074 次；原机位 MAE 改善约 1.9%，但留出视角 MAE 略差，未通过
全机位保真门槛，未替换已有单一 PLY，也未作为本次 Mesh 的输入。

## 复现

GPU 使用原有 Windows 包装脚本，Open3D 使用独立 mesh-o3d-env，贴图与
深度过滤使用原有 CPU 环境。照片 / 模型文件不包含在仓库中。

```powershell
.\run-refit-gpu.ps1 scene_fusion/single_gs_mesh.py capture --source four-room/scenes/geometry-quads-v1 --model four-room/scenes/tour-appearance-six-v1/optimized.npz --out four-room/scenes/geometry-smooth-depth-raw-six-v1 --size 768
python scene_fusion/depth_geometry_smooth.py --source four-room/scenes/geometry-smooth-depth-raw-six-v1 --out four-room/scenes/geometry-smooth-depth-strong-six-v1 --profile strong
python scene_fusion/single_gs_mesh.py integrate --source four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/geometry-smooth-mesh-strong-six-v1 --resolution 512 --max-faces 1200000
python scene_fusion/mesh_surface_repair.py --mesh four-room/scenes/geometry-smooth-mesh-strong-six-v1/surface.npz --cameras four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/geometry-smooth-repair-six-v1
python scene_fusion/poisson_surface.py --mesh four-room/scenes/geometry-smooth-repair-six-v1/surface.npz --cameras four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/geometry-smooth-guide-poisson-six-v1 --depth 9 --samples 500000 --max-faces 1200000 --point-smooth-iterations 4
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-smooth-depth-strong-six-v1 --geometry four-room/scenes/geometry-smooth-guide-poisson-six-v1/surface.npz --out four-room/scenes/geometry-smooth-guide-photo-six-v1 --warped --footprint --texture-width 7680 --smoothing 0 --visibility-cache four-room/scenes/geometry-smooth-guide-poisson-six-v1/visibility.npz
```

其他可选诊断：`depth_geometry_cloud.py` 导出原机位平滑深度点云；
`depth_source_support.py` 将虚拟深度投影回邻近原机位，分别处理一致、
遮挡未知和自由空间冲突；`mesh_plane_quality.py` 比较估计平面邻域。
这些不会自动使输入变成可靠的实测几何。

## 参考与实现边界

- [DN-Splatter / AGS-Mesh](https://github.com/maturk/dn-splatter) 将深度和法线
  约束用于高斯及 Mesh；其 [导出代码](https://github.com/maturk/dn-splatter/blob/main/dn_splatter/export_mesh.py)
  展示了深度 / 法线回投为有向点云、过滤并进行 Poisson 的路线。本项目没有
  引入整套 DN-Splatter 训练器，也没有把本次方法称为 AGS-Mesh。
- [Open3D 表面重建](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
  是本项目实际使用的 Screened Poisson 实现接口。
- 双边点 / 法线处理为本项目独立实现，位置、方向和累计位移均有保护；
  对弱约束曲面仍可能收缩或变形。验证记录见对应 experiments 目录。
