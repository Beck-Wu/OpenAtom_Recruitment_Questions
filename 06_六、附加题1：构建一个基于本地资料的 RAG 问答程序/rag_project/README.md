# 基于本地资料的 RAG 问答程序

本项目使用 DeepSeek 大模型 API，实现一个基于本地 Markdown 资料的 RAG 问答程序。程序会读取本地知识库，切分文本、建立向量索引，检索与问题最相关的片段，并让大模型基于这些资料生成回答。

## 环境要求

- Python 3.8+
- DeepSeek API Key

## 安装依赖

```bash
pip install -r requirements.txt
```
## 配置密钥
复制 .env.example 为 .env，填入你的 DeepSeek 密钥：  
DEEPSEEK_API_KEY=你的密钥  
.env 已被 .gitignore 忽略，不会提交到 GitHub。

## 运行
python main.py  
所用模型与平台  
Embedding：shibing624/text2vec-base-chinese

大模型：DeepSeek（https://api.deepseek.com）
## 项目结构
├── knowledge_base/       # 本地知识库（Markdown 文件）  
├── main.py               # RAG 主程序  
├── requirements.txt      # 依赖  
├── .env.example          # 密钥变量名示例  
├── .gitignore            # 忽略规则  
└── README.md             # 说明文档  
## RAG 流程
读取 knowledge_base/ 下的 Markdown 文档

按段落切分成文本块

用 Embedding 模型把文本块转成向量，建立索引

用户提问时，把问题转成向量，检索最相关的 Top-K 块

把检索结果和问题交给 DeepSeek，生成回答

回答末尾标注参考来源