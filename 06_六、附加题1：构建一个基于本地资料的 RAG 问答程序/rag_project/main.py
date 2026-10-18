import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os
import glob
import re
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置参数
KB_DIR = "knowledge_base"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"  # 推荐的轻量中文 Embedding 模型
CHUNK_SIZE = 400  # 每个文本块字符数
CHUNK_OVERLAP = 80  # 相邻文本块重叠字符数
TOP_K = 3  # 检索最相关的文本块数量


class SimpleRAG:
    def __init__(self):
        print("正在加载 Embedding 模型...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 初始化 OpenAI 客户端（兼容兼容 OpenAI 格式的第三方或官泄 API）
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com")
        )
        self.model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

        self.documents = []  # 存储原文或文本块元数据
        self.chunks = []  # 存储切分后的文本块及来源
        self.chunk_embeddings = None

    def load_documents(self):
        """1. 加载本地文档"""
        self.documents = []
        pattern = os.path.join(KB_DIR, "*.md")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"未在 {KB_DIR} 目录下找到任何 Markdown 文件！")

        for filepath in sorted(files):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                self.documents.append({"filename": filename, "content": content})
                print(f"已读取文件：{filename}，字符数：{len(content)}")


    def split_documents(self):
        self.chunks = []
        for doc in self.documents:
            filename = doc["filename"]
            content = doc["content"]

            # 按 Markdown 标题切分（# 开头）
            sections = re.split(r"\n(?=#{1,3} )", content)

            for sec in sections:
                sec = sec.strip()
                if not sec:
                    continue
                if len(sec) <= CHUNK_SIZE:
                    self.chunks.append({"filename": filename, "text": sec})
                else:
                    start = 0
                    while start < len(sec):
                        end = start + CHUNK_SIZE
                        self.chunks.append({"filename": filename, "text": sec[start:end]})
                        start += CHUNK_SIZE - CHUNK_OVERLAP

        print(f"文本切分完成！共生成 {len(self.chunks)} 个文本块。")

    def build_index(self):
        """3. 建立向量索引"""
        print("正在将文本块转换为向量并建立索引...")
        texts = [chunk["text"] for chunk in self.chunks]
        # 批量计算 embedding
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        self.chunk_embeddings = np.array(embeddings)
        print("向量索引构建成功！")

    def retrieve(self, question):
        """4. 实现检索，返回最相关的 Top-K 文本块"""
        q_embedding = self.encoder.encode([question], normalize_embeddings=True)[0]

        # 计算余弦相似度（由于已经做过 normalize，直接点积即可）
        similarities = np.dot(self.chunk_embeddings, q_embedding)

        # 获取相似度从高到低的索引
        top_indices = np.argsort(similarities)[::-1][:TOP_K]

        results = []
        for idx in top_indices:
            results.append({
                "filename": self.chunks[idx]["filename"],
                "text": self.chunks[idx]["text"],
                "score": float(similarities[idx])
            })
        return results

    def generate_answer(self, question, retrieved_chunks):
        """5. 调用大模型生成回答"""
        # 整理检索到的文本及来源
        context_str = ""
        used_files = set()
        for i, chunk in enumerate(retrieved_chunks):
            context_str += f"--- 资料片段 {i + 1} (来源: {chunk['filename']}) ---\n{chunk['text']}\n\n"
            used_files.add(chunk['filename'])

        # 构造提示词
        prompt = f"""请严格根据以下提供的参考资料回答用户的问题。
要求：
1. 仅根据提供的资料回答，严禁编造或凭空猜测。
2. 如果无法从资料中得到答案，请明确回答：“资料中未提及”。
3. 在最终回答的末尾列出所有实际参考的来源文件名。

【参考资料】
{context_str}

【用户问题】
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个严谨的基于知识库的问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    def answer(self, question, q_index):
        print("\n" + "=" * 60)
        print(f"【问题 {q_index}】{question}")
        print("=" * 60)

        retrieved_chunks = self.retrieve(question)

        print("\n--- 检索结果 ---")
        for i, rc in enumerate(retrieved_chunks):
            preview = rc['text'].strip().replace("\n", " ")[:120]
            print(f"Top {i + 1} | {rc['filename']} | 相似度 {rc['score']:.4f}")
            print(f"        {preview}...")

        print("\n--- 最终回答 ---")
        answer = self.generate_answer(question, retrieved_chunks)
        print(answer)

        # 统一在末尾列出来源，只显示文件名
        sources = sorted(set(rc['filename'] for rc in retrieved_chunks))
        print("\n参考来源：" + "、".join(sources))
        print("=" * 60 + "\n")

        return answer, retrieved_chunks


if __name__ == "__main__":
    rag = SimpleRAG()
    rag.load_documents()
    rag.split_documents()
    rag.build_index()

    # 测试要求中的四个问题
    test_questions = [
        "本次招新的材料提交截止时间是什么？",
        "API Key 应如何保存，哪些内容不能提交到 GitHub？",
        "没有独立显卡的同学能否完成 RAG 题？应采用什么方式？",
        "本次招新的面试地点和面试时间是什么？"
    ]

    for i, q in enumerate(test_questions, 1):
        rag.answer(q, i)  # ← 传两个：问题 + 序号