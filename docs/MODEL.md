# 图像诊断模型接入（YOLO / ONNX）

**现状**：仓库不附带训练好的权重。找不到模型文件时，`diagnose_leaf_image` 返回**演示数据**
（`source=placeholder`），并在返回内容里明确标注"未部署本地检测模型"。放入模型文件后自动切换为真实推理。

## 两种可插拔后端

| 模型文件 | 依赖（可选） | 说明 |
| --- | --- | --- |
| `models/leaf_disease.pt` | `ultralytics` | YOLO 检测/分割模型，直接 `YOLO(...).predict` |
| `models/leaf_disease.onnx` | `onnxruntime` + `Pillow` | 通用部署格式，CPU 即可推理 |

模型路径可用环境变量 `AGRI_LEAF_MODEL` 覆盖（绝对路径或相对路径均可）。

## ONNX 输入/输出契约（v1）

- 输入：RGB、缩放到 640×640、`/255` 归一化、CHW、NCHW、float32
- 输出：`[1, N, 6]`，每行 `x1, y1, x2, y2, score, class_id`（YOLO 常见导出形态）
- 置信度阈值 0.25，最多返回 50 个目标；类别默认返回 `class_id`，中文病名映射表待接入

导出命令示例（ultralytics）：

```bash
yolo export model=leaf_disease.pt format=onnx opset=12 imgsz=640
```

## 接入真实模型时的验收清单

- [ ] 留出独立验证集，报告 mAP@50 或病斑分割精度（不要只报训练集数字）
- [ ] 补 `models/MODEL_CARD.md`：数据来源、标注规范、已知失败案例（逆光、早期病斑、远距离）
- [ ] 低置信度兜底话术：图像不清晰时明确说明"建议补拍或人工复核"，不要硬给结论
- [ ] 与知识库结论冲突时优先提示人工复核——农业场景的错误建议代价高
