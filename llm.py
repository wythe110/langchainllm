# 导入必要的库
# LangChain的Ollama集成模块，用于与Ollama服务交互
from langchain_ollama import OllamaLLM
# LangChain的提示模板，用于格式化输入给LLM的提示
from langchain_core.prompts import ChatPromptTemplate

def main():
    # 配置Ollama服务和模型
    # model参数指定要使用的模型，这里是Qwen3:14b
    # base_url参数指定Ollama服务的地址，这里是你提供的10.86.5.43:11434
    llm = OllamaLLM(
        model="Qwen3:14b",
        base_url="http://10.86.5.43:11434"
    )
    
    # 创建一个提示模板
    # 模板中{question}是一个变量，后续会被实际问题替换
    prompt = ChatPromptTemplate.from_template("请回答以下问题：{question}")
    
    # 将提示模板和LLM模型组合成一个链
    # 链(chain)是LangChain中处理流程的基本单元，这里的链会将提示传递给LLM
    chain = prompt | llm
    
    # 定义一个问题
    question = "什么是人工智能？请用简单易懂的语言解释"
    
    # 调用链来获取回答
    # invoke方法用于执行链，参数是一个字典，包含要替换的变量
    response = chain.invoke({"question": question})
    
    # 打印问题和回答
    print(f"问题：{question}")
    print(f"\n回答：{response}")

# 如果这个脚本是直接运行的（而不是被导入的），则执行main函数
if __name__ == "__main__":
    main()

#zheyang