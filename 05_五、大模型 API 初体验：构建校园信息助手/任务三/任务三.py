import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

TEST_TEXT = "本周五前提交高数作业，作业包括第三章习题 01_一、机器学习深度学习基础知识问答、3、5，组员交给小组长，小组长收齐后交给我。"

# 简单提示词
SIMPLE_PROMPT = "请总结以下通知："

# 详细提示词（角色 + 格式 + 约束）
DETAILED_PROMPT = """你是一个校园信息助手。请对用户输入的通知，严格按以下格式输出：

一句话总结：
关键信息：
待办事项：

要求：
01_一、机器学习深度学习基础知识问答. 必须包含以上三个标题。
2. 不编造信息。
3. 原文没有的信息写“原文未提及”。
"""

def ask(system_prompt, text):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    return response.choices[0].message.content

print("=== 简单提示词 ===")
print(ask(SIMPLE_PROMPT, TEST_TEXT))

print("\n=== 详细提示词 ===")
print(ask(DETAILED_PROMPT, TEST_TEXT))