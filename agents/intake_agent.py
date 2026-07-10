"""
信息采集向导模块
在投诉信起草前，引导消费者提供完整的纠纷信息
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import ARK_API_KEY, ARK_BASE_URL, MODEL_ID


INTAKE_SYSTEM_PROMPT = """你是一个消费维权信息采集向导。你的职责是通过对话引导消费者提供完整的纠纷信息，为后续起草投诉信做准备。

你需要采集以下关键信息:
1. 消费者姓名和联系方式
2. 商家名称和地址（或电商平台名称）
3. 购买时间
4. 商品或服务名称
5. 购买金额
6. 问题描述（发生了什么纠纷）
7. 与商家沟通的过程和结果
8. 消费者的诉求（退款、赔偿、道歉等）

对话规则:
- 一次只问一到两个问题，不要一次性问太多
- 根据用户的回答追问缺失的关键信息
- 语气友善、耐心，让消费者感到被理解
- 当所有关键信息收集完毕后，总结信息并告知用户可以开始起草投诉信
- 如果用户想跳过某些信息，尊重用户的意愿
- 不要编造用户没有提供的信息"""


# 关键信息字段
REQUIRED_FIELDS = {
    "consumer_name": "消费者姓名",
    "consumer_contact": "联系方式",
    "merchant_name": "商家名称",
    "purchase_time": "购买时间",
    "product_name": "商品或服务名称",
    "amount": "购买金额",
    "issue": "问题描述",
    "demand": "诉求",
}


class IntakeAgent:
    """
    信息采集向导
    通过多轮对话收集消费纠纷信息
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_ID,
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            temperature=0.5,
            streaming=True,
        )
        self.system_prompt = INTAKE_SYSTEM_PROMPT
        self.collected_info = {}  # 已采集的信息

    async def chat(self, user_input: str, history: list = None):
        """
        流式对话，采集信息

        生成器:
            逐步 yield 回复文本
        """
        messages = [SystemMessage(content=self.system_prompt)]

        # 加入对话历史
        if history:
            for h in history[-6:]:  # 保留最近6轮
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h.get("role") == "assistant":
                    messages.append(SystemMessage(content=h["content"]))

        messages.append(HumanMessage(content=user_input))

        full_response = ""
        try:
            async for event in self.llm.astream_events(messages, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        full_response += chunk
                        yield full_response
        except Exception as e:
            yield f"抱歉，处理时出错: {str(e)}"

