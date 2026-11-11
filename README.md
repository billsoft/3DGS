# 3DGS — 多全景约束的三维表面融合

2026-11-11 用户验收的新基线。针对四个机位的 HY-World 全景图、Gaussian PLY 和辅助 collider 数据，优化公共表面、划分三维照片领地，导出融合 PLY、SPLAT 和贴图网格。

这是当前会议室实验的可追溯代码快照，**不是任意四张图上传后就能直接运行的通用产品**。当前阶段需要已有相机标定、深度缓存及 V3 初始化表面；这些私有数据不随公开仓库发布。原始 PLY 未参与端到端可微联合训练，最终高斯由优化后的表面重新采样生成。

## 算法

1. 深度重投影后的双向 DIS 光流，检查遮挡、双向一致性和三角化误差。
2. 以空间块留出对应点验证表面法向位移；平滑正则、位移上限和翻转冻结保护几何。
3. 对公共表面做一致细分、射线可见性检查，以 alpha-expansion / max-flow 划分四个照片来源与未知区域。
4. 区域内部单一来源，只在表面邻接的窄接缝带中，对颜色、深度和对应点一致的样本羽化。
5. 受限照片 RGB 增益/偏置校正，再沿表面传播受限接缝颜色偏置。
6. 弱证据且与有效初值冲突时保留原贴图；删除无依据的灰色占位面，重新采样表面高斯。

新优化层无灯具专用模板；初始化仍含此前电视、球体、桌面拟合先验。灯具锯齿、背面缺失、桌底空洞尚未解决。删除错误几何可能增大空洞。网格贴图预览不含 PLY 的局部颜色偏置和窄带羽化。

## Windows 运行

验证环境：原生 Windows、Python 3.12.10。无需 WSL。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:GS_INPUT_DIR = 'D:\your-room-input'
.\.venv\Scripts\python.exe fusion_next/pipeline.py
# 同一份输入和标定重复实验时，才允许使用缓存：
.\.venv\Scripts\python.exe fusion_next/pipeline.py --reuse-correspondences
.\.venv\Scripts\python.exe server.py
```

打开 http://127.0.0.1:7870/fusion-next-room.html 。服务仅监听本机。W/S 前后、A/D 左右平移、Q 上升、E 下降。页面提供四个机位、四个中点、V3 对照、领地图及验证报告。

未恢复下列数据时，请先运行 `python check_data.py` 查看缺失项；代码不会自动下载或猜测这些标定数据。

## 私有输入与已有处理产物

`GS_INPUT_DIR` 下放 `2_7680.jpg` 至 `5_7680.jpg` 和对应 `2.ply` 至 `5.ply`。照片为完整等距柱状全景，本次原片尺寸 7680×3840。站点编号及检查视点沿用当前四机位实验。

仓库根目录需要恢复以下 `four-room/` 数据：

| 路径 | 内容 |
| --- | --- |
| `registration.json`, `fusion.json` | 前期相似变换、站点、相机中心及 up |
| `{2,3,4,5}/cache.npz` | 原始 PLY 相机坐标标定 R、C |
| `{2,3,4,5}/depth-{0..5}.npz` | 原始 Gaussian 六面渲染 depth、alpha |
| `colliders/{2..5}-{0..5}.npy` | collider 辅助深度 |
| `photo-bundle/bundle.npz` | 照片联合标定后的 C、Q |
| `photo-bundle/depth-report.json` | 各来源深度尺度修正 |
| `surface-fusion-v3/mesh.bin`, `mesh.json`, 全部引用贴图、`fused.ply`、`fused.splat`、`fusion.json` | V3 初始化与对照 |
| `surface-fusion-v4/inspection-views.json` | 固定诊断视点；只用于验收渲染，不用于几何优化 |

从原始输入重新生成这些历史产物的完整自动化不在本基线入口内。不要把生成深度当作真实扫描深度。不同数据不能复用旧对应点缓存。

## 结果与复现记录

`fusion_next/pipeline.py` 顺序运行对应、曝光、几何、归属、接缝校色、导出、近景和中点验证、网格打包及页面发布。输出在 `four-room/fusion-next/`；重复运行会覆盖该输出目录，先另存已接受的基线。

基线的数值报告与模型 SHA-256 在 `baselines/2026-11-11/`。输出 3,166,125 个高斯，23,208 个参与边界混色。留出几何法向残差中位数从 0.034385 降至 0.025490（模型单位），翻转面片为 0。此指标来自照片对应，不是真实深度精度，也不是画质提升百分比。

原照片、模型、纹理和渲染截图均不提交。公开代码中的标定格式和页面针对当前房间；复现相同视觉结果需要同一份私有数据。用户认可的是这一实验结果，不代表所有区域已修复。

## 通用保边细节试验

`fusion_next/run_detail.py` 使用已接受基线的几何和区域归属，按原图梯度及投影像素跨度分配采样，在边缘法向收缩高斯足迹，并只融合接缝低频颜色。没有灯具或窗帘专用贴图、分割区域或合成形状。

先将已接受输出目录完整另存到 `four-room/baselines/2026-11-11-fusion-next/`。这份私有缓存需包含 `geometry.npz`、`ownership.npz`、`prior.npz`、`correspondences.npz`、`seam-offset.npy`、各阶段 JSON 报告以及基线 `fused.ply` / `fused.splat`。然后在上述 Python 环境与 `GS_INPUT_DIR` 设置下运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe fusion_next/run_detail.py --validate --serve-root .
```

结果写入独立的 `four-room/fusion-detail/`，预览地址为 `http://127.0.0.1:7870/fusion-detail-room.html`；已接受的基线入口保持不变。可用 `--smooth` 显式试验保边法向平滑，但默认关闭：消融试验没有发现值得默认改变基线几何的收益。平滑无法修复缺失表面和错误拓扑，边缘优化也无法保证消除所有灰色污染。

验证包含固定近景、四个中点、7 个原图来源视向；后者只在共同高覆盖率像素上计算 SSIM 和 MAE，不将其解释为所有漫游视点的画质。每来源采样预算上限为 1.8 倍；高斯数量增加会提高浏览器内存与加载成本。

## 第三方组件与许可

- `three.js`：Three.js r160 / 0.160.0，MIT。
- `gaussian.js`：GaussianSplats3D 0.4.7，MIT；文件与官方 npm 发布包一致。
- `mesh-orbit-controls.js`：Three.js OrbitControls 衍生文件。

第三方许可见 `third_party/`。本仓库未额外指定项目代码的开源许可；公开可见不等同于授权任意再分发。
