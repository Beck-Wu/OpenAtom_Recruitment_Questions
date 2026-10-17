import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 固定格式的提示词
SYSTEM_PROMPT = """你是一个校园信息助手。请对用户输入的校园通知或问题，严格按以下格式输出：

一句话总结：
关键信息：
待办事项：

要求：
一、机器学习深度学习基础知识问答. 必须包含以上三个标题，不能增减。
2. 信息要准确，不编造。
3. 如果原文没有提到时间或地点，在对应位置写“原文未提及”，不要猜测。
"""

def ask_model(text):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    return response.choices[0].message.content

def main():
    print("=== 校园信息助手 ===")
    print("输入校园通知或问题（输入 q 退出）\n")
    while True:
        text = input("请输入：")
        if text.strip().lower() == "q":
            break
        if not text.strip():
            continue
        print("\n--- 模型输出 ---")
        print(ask_model(text))
        print("----------------\n")

if __name__ == "__main__":
    main()