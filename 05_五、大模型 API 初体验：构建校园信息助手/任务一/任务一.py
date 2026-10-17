import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": "你是一个校园信息助手。"},
        {"role": "user", "content": "学校图书馆周末开放吗？"}
    ],
    stream=False
)

print(response.choices[0].message.content)