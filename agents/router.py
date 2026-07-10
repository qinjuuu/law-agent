"""
意图路由器模块
通过 LLM 分析用户输入，将请求分发到对应的专业智能体
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import ARK_API_KEY, ARK_BASE_URL, MODEL_ID


# 意图分类
INTENT_COMPLAINT = "complaint"        # 投诉信起草
INTENT_REVIEW = "review"              # 格式条款审查
INTENT_QA = "qa"                      # 消费法律问答


ROUTER_SYSTEM_PROMPT = """你是一个意图识别助手。你的任务是分析用户的输入，判断其意图属于以下哪一类:

1. complaint - 投诉信起草: 用户想写投诉信、举报信、维权函，需要帮助生成正式的维权文书
2. review - 格式条款审查: 用户想审查合同条款、消费协议、霸王条款，需要分析条款的合法性和风险
3. qa - 消费法律问答: 用户咨询消费者权益相关问题，想了解自己的权利、维权方式、法律规定

请只输出一个单词: complaint、review 或 qa，不要输出其他内容。"""


class IntentRouter:
    """
    意图路由器
    使用 LLM 对用户输入进行意图分类，返回对应意图标签
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_ID,
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            temperature=0.0,  # 路由需要确定性输出
        )

    def classify(self, user_input: str) -> str:
        """
        对用户输入进行意图分类

        参数:
            user_input: 用户输入文本

        返回:
            意图标签: complaint / review / qa / unknown
        """
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        try:
            response = self.llm.invoke(messages)
            result = response.content.strip().lower()
            # 清理可能的多余内容
            for intent in [INTENT_COMPLAINT, INTENT_REVIEW, INTENT_QA]:
                if intent in result:
                    return intent
            return INTENT_QA  # 默认走问答
        except Exception as e:
            print(f"[Router] 意图识别失败: {e}")
            return INTENT_QA  # 出错时默认走问答

