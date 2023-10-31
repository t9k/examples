# AutoGen 应用示例

[AutoGen](https://github.com/microsoft/autogen) 是一个开发 LLM 应用的框架，这些应用通过多智能体彼此对话以解决任务。AutoGen 智能体是可定制的、可对话的，同时也能让人类无缝参与。它们可以以多种模式运行，利用 LLM、人类输入和工具的组合来完成任务。

## 使用方法

您可以在任何 Jupyter 环境（例如 Jupyter Notebook、Jupyter Lab、VSCode）下运行各个应用示例的 `.ipynb` 文件。

若使用 OpenAI 提供的 GPT 系列模型 API 作为 LLM 后端，请在 `OAI_CONFIG_LIST` 文件中提供您的 API Key；若使用 FastChat 本地部署的开源模型作为 LLM 后端，请参照[官方教程](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)启动服务，并在 `OAI_CONFIG_LIST` 文件中相应地修改配置。最后，在各个 `.ipynb` 文件的“Set your API Endpoint”部分指定您所使用模型的名称。

## 解数学题

本示例使用 [MathChat](https://arxiv.org/abs/2306.01337) 框架来解答数学问题，修改自官方教程 [Auto Generated Agent Chat: Using MathChat to Solve Math Problems
](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb)。

本示例需要安装 `pyautogen[mathchat]`：

```bash
pip install "pyautogen[mathchat]~=0.1.1"
```

然后运行 `MathChat.ipynb` 文件的全部单元格。

## 文档问答

本示例是一个检索增强的问答应用，修改自官方教程 [Auto Generated Agent Chat: Using RetrieveChat for Retrieve Augmented Code Generation and Question Answering](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb)。

本示例需要安装 `pyautogen[retrievechat]` 和 `flaml[automl]`：

```bash
pip install "pyautogen[retrievechat]~=0.1.2" "flaml[automl]"
```

将提供上下文的文档文件下载到当前目录下：

```bash
wget https://huggingface.co/datasets/thinkall/NaturalQuestionsQA/resolve/main/corpus.txt
wget https://www.szse.cn/api/disc/info/download?id=f44320dc-1129-44aa-9924-a2d212f16e70 -O byd_financial_report_2023h1.pdf
```

然后运行 `RetrieveChat.ipynb` 文件的全部单元格。

## 可视化数据（群聊）

在本示例中，多智能体通过动态群聊的方法编写代码以可视化表格数据，本示例修改自官方教程 [Auto Generated Agent Chat: Group Chat with Coder and Visualization Critic](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_vis.ipynb)。

直接运行 `groupchat_visualization.ipynb` 文件的全部单元格。
