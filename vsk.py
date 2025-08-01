from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
import base64
import io
from PIL import Image
import os

def image_to_base64(image_path):
    """将图片转换为base64编码，支持JPG和PNG格式"""
    try:
        with Image.open(image_path) as img:
            # 处理透明通道（PNG可能有alpha通道）
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # 创建白色背景
                background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                background.paste(img, img.split()[-1])
                img = background
            
            # 转换为RGB格式确保兼容性
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 压缩图片，避免base64字符串过长
            img.thumbnail((1024, 1024))  # 适当增大尺寸以保证识别准确性
            buffer = io.BytesIO()
            
            # 根据原文件扩展名选择保存格式
            if image_path.lower().endswith('.png'):
                img.save(buffer, format='PNG')
                mime_type = 'image/png'
            else:  # 默认JPG
                img.save(buffer, format='JPEG', quality=90)
                mime_type = 'image/jpeg'
                
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:{mime_type};base64,{img_base64}"  # 完整的data URI格式
    except Exception as e:
        print(f"图片处理错误: {e}")
        return None


def identify_animal(image_path):
    # 1. 转换图片为base64 (带MIME类型的完整格式)
    img_data_uri = image_to_base64(image_path)
    if not img_data_uri:
        return "图片处理失败，请检查路径或格式"

    # 2. 配置Ollama连接
    llm = OllamaLLM(
        base_url="http://10.86.5.43:11434",
        model="qwen2.5vl:7b",
        temperature=0.1
    )

    # 3. 使用HumanMessage格式传递多模态信息（文本+图片）
    message = HumanMessage(
        content=[
            {"type": "text", "text": """请仔细分析图片中的动物，回答：
1. 这是什么动物？
"""},
            {"type": "image_url", "image_url": {"url": img_data_uri}}
        ]
    )

    # 4. 调用模型
    try:
        response = llm.invoke([message])
        return response
    except Exception as e:
        return f"模型调用失败: {e}"


if __name__ == "__main__":
    animal_image_path = "cat.jpg"  # 可以是JPG或PNG格式
    result = identify_animal(animal_image_path)
    print("识别结果：")
    print(result)


print("图片是否存在：", os.path.exists(animal_image_path))  # 会打印True或False