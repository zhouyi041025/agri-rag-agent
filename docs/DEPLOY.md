# 部署到公网（在线 Demo）

目标：得到一个**面试官在中国大陆能直接打开**的链接，点进去就能对话。

## 零、先看这张表：哪些平台在国内打得开

2026-09-14 在本地网络实测（超时 10 秒）：

| 平台 | 可达性 | 结论 |
| --- | --- | --- |
| ModelScope 魔搭社区 | ✅ 204 ms | **推荐**，创空间免费、国内直连 |
| `*.vercel.app` 应用域名 | ❌ 超时 | 不建议，链接发出去对方打不开 |
| Hugging Face / Spaces | ❌ 超时 | 不建议，国内不可达 |
| Render | ❌ 超时 | 不建议 |
| Zeabur / Sealos | ✅ 1.0 s / 0.2 s | 备选，需要付费或实名 |

本项目的运行环境很轻：不下载模型、不需要 GPU、不需要 API Key，
所以免费档的 CPU 实例完全够用。

---

## 一、方案 A：魔搭创空间（推荐）

### 1. 准备账号

打开 <https://www.modelscope.cn> 用阿里云账号登录，完成实名认证。

### 2. 创建创空间

1. 右上角头像 → **创空间** → **创建创空间**
2. 填写名称（例如 `agri-rag-agent`）与简介
3. **代码框架 / SDK 类型**这一栏：
   - 如果有 **Docker**，选它 —— 本仓库根目录已经有 `Dockerfile`，开箱即用
   - 如果只有 **Gradio / Streamlit**，先别选，告诉我，我给你补一个 Gradio 入口
4. 可见性选**公开**

### 3. 把代码推上去

创建完成后页面会给一个 Git 地址，形如
`https://www.modelscope.cn/studios/<你的用户名>/agri-rag-agent.git`。

在项目根目录执行（把地址换成你自己的）：

```powershell
git init
git add .
git commit -m "荟诊：芦荟病虫害智能问答与诊断 Agent"
git remote add origin https://www.modelscope.cn/studios/<你的用户名>/agri-rag-agent.git
git push -u origin master
```

推送时会要求登录，用魔搭的账号密码或 Access Token 即可。

### 4. 等待构建

创空间页面会显示构建日志。首次构建需要装依赖，大约 1-3 分钟。
构建完成后，页面右上角的链接就是可以发出去的 Demo 地址。

### 5. 自检

打开 Demo，确认：

- 右上角状态栏显示「规则降级模式（未配置 API Key）· 知识块 29」
- 点一下输入框上方的示例问题，能出答案、右侧有引用和工具轨迹

---

## 二、方案 B：任意容器平台

仓库根目录的 `Dockerfile` 是通用的，Zeabur、Sealos、Railway 或自己的服务器都能直接用：

```bash
docker build -t agri-rag-agent .
docker run -p 7860:7860 agri-rag-agent
```

服务监听 `7860`，也支持平台注入的 `PORT` 环境变量，可以按需覆盖：

```bash
docker run -p 8080:8080 -e PORT=8080 agri-rag-agent
```

---

## 三、方案 C：面试当天的本机演示（应急）

如果临时需要给对方看，又不想等部署：

```powershell
.\run.ps1          # 本机启动，浏览器打开 http://127.0.0.1:7860
```

配合内网穿透工具（cpolar、花生壳、ngrok 等）可以把本机端口映射成一个临时公网地址。
注意：这种方式依赖你的电脑开着，只适合面试当下演示，不要写进简历。

---

## 四、想让它更聪明？（可选）

默认是**离线规则模式**，不调用任何外部接口，答案是从知识库抽取的原文。
如果想让它是真正的「大模型 Agent」，在创空间的环境变量里加一个 Key 即可：

| 变量 | 值 |
| --- | --- |
| `AGRI_LLM_PROVIDER` | `deepseek` |
| `AGRI_LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `AGRI_LLM_MODEL` | `deepseek-chat` |
| `AGRI_LLM_API_KEY` | 你的 Key |

配好后 Agent 会自动切换为 Function Calling 模式：自主决定调哪个工具、生成自然语言回答，
并保留引用溯源。DeepSeek 的 API 按量计费，一个演示站点一个月通常几块钱。

---

## 五、常见问题

**Q：构建成功但打开是 404 / 502？**
确认容器监听的是 `7860` 端口。`app.py` 已默认读取 `PORT` 环境变量，平台若注入其他端口也能自适应。

**Q：会不会因为没 GPU 跑不起来？**
不会。项目不加载任何深度学习模型，`diagnose_leaf_image` 是预留接口，返回结构化占位结果。

**Q：数据会不会丢？**
不会。田间环境数据在 `data/env/field_env.csv`，索引在容器启动时自动构建，无状态。

**Q：能不能改端口？**
可以，设置环境变量 `PORT` 即可，`app.py` 会优先读取它。
