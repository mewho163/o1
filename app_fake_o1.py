# -*- coding: utf-8 -*-
import streamlit as st
import openai
import os
import json
import time
from dotenv import load_dotenv  # Import dotenv for loading environment variables

# Load environment variables
load_dotenv()

# Get configuration from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_BASE_URL

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            content = response.choices[0].message['content']
            print("DEBUG: API response content:", content)
            return json.loads(content) if content.startswith('{') else {"title": "Error", "content": content}
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error",
                            "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """您是一个具有高级推理能力的AI助手。您的任务是详细解释您的思维过程的每一步。对于每一步：
        1. 提供描述当前推理阶段的明确、简洁的标题。
        2. 在内容部分详细说明您的思维过程。
        3. 决定是继续推理还是提供最终答案。
        响应格式：
        使用带有键的JSON：'title'，'content'，'next_action'（值：'continue'或'final_answer'）
        关键指示：
        - 使用至少5个不同的推理步骤。
        - 承认您作为AI的局限性并明确说明您能做什么和不能做什么。
        - 积极探索和评估替代答案或方法。
        - 批判性地评估自己的推理；识别潜在的缺陷或偏见。
        - 在重新审视时，使用根本不同的方法或视角。
        - 利用至少3种不同的方法来得出或验证您的答案。
        - 在推理中结合相关领域知识和最佳实践。
        - 在每一步和最终结论中定量确定水平时适用。
        - 考虑推理的潜在边缘情况或例外情况。
        - 为排除替代假设提供清晰的理由。
        有效JSON响应示例：
        {
            "title": "初步问题分析",
            "content": "为了有效地解决这个问题，我将首先将给定的信息分解为关键组成部分。这涉及...[详细说明]...通过这种方式结构化问题，我们可以系统地解决每个方面。",
            "next_action": "continue"
        }""" },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "谢谢！我现在将按照我的指示逐步思考，从分解问题开始。"}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        # Ensure step_data is a dictionary
        if not isinstance(step_data, dict):
            step_data = {"title": "Error", "content": str(step_data), "next_action": "final_answer"}

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data.get('next_action') == 'final_answer':
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "请根据上述推理提供最终答案。"})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="OpenAI Reasoning Chains",  layout="wide")

    st.title("使用OpenAI创建推理链")

    st.markdown("""
    这是一个使用OpenAI模型创建推理链的原型，以提高输出准确性。 
    该准确性尚未经过正式评估。

    从[bklieger-groq](https://github.com/bklieger-groq)分叉
    开源[repository here](https://github.com/win4r/o1)
    """)

    # Text input for user query
    user_query = st.text_input("输入您的问题：", placeholder="例如：草莓这个词里有几个 R？")

    if user_query:
        st.write("正在生成响应...")

        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)

            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**总思考时间：{total_thinking_time:.2f} 秒**")

if __name__ == "__main__":
    main()
