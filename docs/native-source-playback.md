# 原始机位保真播放器

目标是在任一标定拍摄点恢复该机位原始 PLY 的渲染结果。此实现是多模型播放，不是新的单一融合 PLY，也不修复原始 PLY 自身的错误。

## 运行

先完成通用流程的单源标定、联合定位和数据集生成，并准备一个用于机位之间的融合基线。然后：

```powershell
python scene_fusion/native_package.py --scene four-room/scenes/general-six-v2 --baseline four-room/scenes/view-color-sh2-v1 --out four-room/scenes/native-source-six-v1
python server.py
```

打开 `/native-scene-room.html?scene=native-source-six-v1`。`--scene` 提供 `scene.json`、`aligned-cameras.npz`、各来源的归一化信息和 `data-512/dataset.json`；`--baseline` 提供 `fusion.json` 与导出的模型。站点数量和按钮来自元数据，没有固定六站逻辑。现有标定输入限定为 degree-zero Gaussian PLY，仍遵守通用运行器的格式要求。

## 保留原始表示

1. 校验每份源文件与标定时的 SHA-256 一致，逐字节复制，不裁点、不重新附色、不重采样。
2. 将全局相机通过原始 PLY 到公共坐标系的相似变换的逆变换，转换到对应模型坐标系。同步位置、朝向、近远裁面；保留原始高斯位置、协方差、透明度与颜色。
3. 每一时刻只显示一个原始来源。拍摄光心的原始层权重严格为 1，融合层不会混入其画面。拖动只转头，不移动光心。
4. 原始模型保护半径是最近相邻机位间距的 8%；在 8%–24% 间用 smoothstep 逐渐过渡到融合模型。相邻原始模型的保护区不交叉。这个屏幕空间过渡可能出现双影，不是经过几何验证的三维融合。

“融合对照”在同一相机位置显示旧融合模型。“单份原始对照”重新加载且只加载当前一份 PLY，用来排查多模型容器引入的差异。对照链接默认水平视向，可用上下检查按钮比较。

## 验证

- `node --test tests/test_native_camera.mjs tests/test_room_look.mjs tests/test_room_sorting.mjs`：相似变换的投影不变性、多种站点数量的权重、原地转头与排序完成性。
- `node tests/check_native_scene.mjs`：在已有专用 CDP 浏览器中检查六机位 × 六方向、单源可见性、位置、图层权重、排序，以及中点和鼠标操作。不会自行启动浏览器。
- `node tests/check_native_reference.mjs`：独立加载单个原始文件，采集同分辨率、同视点的参考图，用于逐像素比较。

不能据文件一致就声称与全景照片一样清晰。实际检查中，1 号源 PLY 未经融合时的天花板和地面也存在细小斑点，而照片更干净。原始相机位置来自照片/PLY 特征标定；它有误差，并非模型生产方提供的精确相机参数。中间区域沿用原融合结果，未观察到的背面仍可能错误。
