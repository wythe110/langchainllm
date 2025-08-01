# 导入必要的库
from langchain_community.llms import Ollama  # LangChain对接Ollama的模块
from langchain_core.prompts import PromptTemplate  # 用于构建提示词模板
from PIL import Image  # 处理图片的库
import base64  # 用于图片base64编码（多模态模型需要）
import io  # 处理字节流


def image_to_base64(image_path):
    """
    将图片转换为base64编码字符串（多模态模型输入图片的常用格式）
    :param image_path: 图片文件路径
    :return: base64编码的字符串
    """
    try:
        # 打开图片并转换为RGB格式（确保兼容性）
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')  # 统一转换为RGB格式
            
            # 将图片保存到字节流（模拟文件读写，避免临时文件）
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')  # 以JPEG格式保存
            
            # 转换为base64编码并解码为字符串
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"图片处理错误: {e}")
        return None


def identify_animal(image_path):
    """
    调用Ollama的qwen2.5vl模型识别动物图片
    :param image_path: 动物图片路径
    :return: 模型识别结果
    """
    # 1. 将图片转换为base64编码（qwen2.5vl需要这种格式输入图片）
    img_base64 = image_to_base64(image_path)
    if not img_base64:
        return "图片处理失败，请检查图片路径或格式"

    # 2. 配置Ollama连接（指定服务器地址和模型）
    llm = Ollama(
        base_url="http://10.86.5.43:11434",  # Ollama服务器地址
        model="qwen2.5vl:32b",  # 使用的模型名称
        temperature=0.1  # 温度参数（0-1，越低结果越确定）
    )

    # 3. 构建提示词模板（告诉模型需要做什么）
    prompt_template = PromptTemplate(
        input_variables=["image_data"],  # 输入变量：图片的base64数据
        template="""请分析以下图片中的动物，告诉我这是什么动物。
如果能确定，简要描述它的特征；如果不确定，说明可能的动物类别。
图片数据：{image_data}
"""
    )

    # 4. 格式化提示词（将图片数据传入模板）
    prompt = prompt_template.format(image_data=img_base64)

    # 5. 调用模型获取结果
    try:
        response = llm.invoke(prompt)  # 发送请求给Ollama服务器
        return response
    except Exception as e:
        return f"模型调用失败: {e}"


if __name__ == "__main__":
    # 图片路径（替换为你的动物图片路径）
    animal_image_path = "animal.jpg"
    
    # 调用识别函数并打印结果
    result = identify_animal(animal_image_path)
    print("识别结果：")
    print(result)