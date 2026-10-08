# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.4.0] - 2026-10-08

### Fixed
- LLM 调用从单次 POST 改为指数退避重试（默认 2 次、上限 8s），网络抖动与 429 不再直接打到用户
- `KnowledgeBase.search()` 读模块级 `default_settings.rrf_k`、忽略传入配置的问题
- 命中查询缓存时就地改 `cached` 标记、污染调用方旧对象的别名问题（现在返回副本）

### Added
- 进程内查询缓存（`AGRI_CACHE_SIZE`，默认 128）与 `GET /stats`（请求、缓存命中、工具/模型调用、token、可选成本）
- 图像诊断可插拔：支持 `YOLO(.pt)` 与 `ONNX(.onnx)`，模型路径 `AGRI_LEAF_MODEL`；契约见 `docs/MODEL.md`
- 同义词表外置为 `data/synonyms.json`；`docker-compose.yml` 一键起服务
- `/chat` 输入上限（2000 字 / 20 条历史）
- LLMAgent 工具循环的离线单测（ScriptedLLM：工具路由、步数上限、失败回填、引用裁剪）

### Changed
- 测试 38 → 59；CI 增加 ruff 与覆盖率门槛（78%，当前 81%）

## [0.3.0] - 2026-09-16

- 多轮追问补全（省略式提问结合上一轮检索意图）、GitHub Actions 自动测试（3.9 / 3.12）、国内镜像构建加固

## [0.2.0] - 2026-09-15

- 引用台账（CitationLedger）与"幽灵引用"裁剪、SSE 事件流、在线体验部署

## [0.1.0] - 2026-09-15

- 初始版本：RAG + Function Calling 芦荟病虫害问答 Agent
