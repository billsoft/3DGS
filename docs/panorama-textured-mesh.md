# 多全景照片贴图网格实验

预览：`/mesh-photo-room.html?scene=mesh-photo-six-v4`。页面包含六个原机位、六个中点、上下视向、原片贴图、法线几何检查、照片负责区域和三角网格显示。下载是实际 OBJ + MTL + 六张烘焙贴图，不是高斯文件改扩展名。

## 输入和处理

脚本读取当前场景的 `scene.json`、`aligned-cameras.npz`、六面预测深度、原全景照片和已重建的 `surface.npz`。来源数量从清单读取；不假设固定四张或六张。新场景仍需先完成标定和几何重建，并非只运行贴图步骤就能得到正确几何。

1. `surface_volume.py` 从同一坐标系中的深度证据提取公共三角面。本次最终试验采用已有多图平面约束修正后的深度。未经修正的深度建网格会保留电视碎层，第一轮实测确认了这一点。
2. `mesh_regularize.py` 清除退化面，只闭合直径不超过局部间距四倍、近似平面的小边界环。拒绝把小孤立组件的外轮廓误当成孔洞。可选择将有多机位约束的有界平面重建为规则面片；没有物体名称、房间坐标或人工灯具贴片。
3. `textured_mesh.py` 做位移受限、保护开边界和尖锐结构的法向平滑；翻转面回退。每张照片从六个方向对实际网格做深度遮挡检查，并结合入射角、投影距离、原深度可信度为三角面评分。
4. 在三角面邻接图上用迭代标签更新选择连续的照片区域。区域内部只取一个主来源，不六图平均；有验证约束的平面可固定一个外观来源，避免反射图像混合。
5. 接缝仅在一圈相邻三角面的范围内羽化。来自第二张图的颜色必须同时满足可见性/深度支持和颜色相似门槛；否则保留主来源。在线性光空间混合后烘焙回 sRGB 贴图，处理全景左右循环接缝。
6. 输出统一网格、六张 4096×2048 贴图、OBJ/MTL 和 ZIP。预览使用无额外光照的贴图材质，不重复叠加照片已有光照。

照片仍携带拍摄时的光照和反射。静态贴图不能同时保证所有机位的镜面反射与原图一致，也不能用颜色羽化修正错位几何。网格存在多个连通分量和开放边界，不是水密扫描模型。

## 复现当前实验

```powershell
python scene_fusion/surface_volume.py --source four-room/scenes/geometry-quads-v1 --warped --out four-room/scenes/mesh-geometry-six-v2 --resolution 360
python scene_fusion/mesh_regularize.py --mesh four-room/scenes/mesh-geometry-six-v2/surface.npz --source four-room/scenes/geometry-quads-v1 --out four-room/scenes/mesh-geometry-six-v4 --planes four-room/scenes/surface-consensus-v1/provisional-quads.json --allow-provisional
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-quads-v1 --geometry four-room/scenes/mesh-geometry-six-v4/surface.npz --warped --out four-room/scenes/mesh-photo-six-v4
python tests/test_textured_mesh.py
node tests/check_photo_mesh.mjs mesh-photo-six-v4
```

上述路径是当前私有测试数据，不随公开代码发布。通用算法不依赖这些具体名称。无平面约束的新场景可省略 `--planes`；不要套用此房间的约束。对新的输入应重新生成标定、深度和网格。

**实验门槛：** 使用的旧平面约束在留出机位上的角点均值误差 3.9523、最大 6.6411 像素，未通过原严格 3.5/6 像素门槛。本轮显式允许已有 5/8 实验门槛，因此始终标为试验，不升级生产基线。角点匹配也不能证明整个平面内部的几何真实。

## 结果和局限

最终 685947 个三角面，六张贴图；闭合 114 个小边界环，增加 228 个补孔三角面。平面约束替换 6124 个原面，插入 5016 个规则面。平滑最大位移约 0.02522 个归一化场景单位，最终没有翻转面；该单位不是米。

第一轮网格的电视碎层严重，墙、门和地面的颜色已经比旧融合外观干净。使用修正深度和规则平面后的版本，电视轮廓及内部完整性明显改善；地面纹理和墙面保持清楚。灯具边缘、桌边、椅子和支架仍有明显破损、缺面及局部投影错位，不能宣称整场景优于原始 PLY 保真漫游。

浏览器检查包括 6 个机位各水平/向上/向下的 18 个视图、6 个中点、几何与来源分区显示；未报告运行或着色器错误。目视检查发现上述残留，因此仅发布独立预览，不替换原始 PLY 入口。单元测试覆盖前后遮挡、透视正确深度、全景 UV 接缝、任意来源数、线性光融合、边界保护、真假孔洞区分。

## 参考实现

- [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing)：研究了其按面来源选择和接缝处理代码。本次采用相同问题分解，但使用自己的简化标签迭代和窄带羽化，没有调用 mapMAP 或复现完整全局接缝求解。
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)：强调面向表面重建的表示与几何约束。本次没有重新训练 2DGS；已有预测深度的不一致仍是主要限制。
