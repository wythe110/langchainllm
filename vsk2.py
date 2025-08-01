import base64
import io
import os
import requests
from PIL import Image

def image_to_base64(image_path):
    """生成纯base64编码（不带data URI前缀）"""
    try:
        with Image.open(image_path) as img:
            # 处理透明通道
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                background.paste(img, img.split()[-1])
                img = background
            
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 压缩图片
            img.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            
            # 保存为JPEG（统一格式减少问题，qwen2.5vl对JPEG兼容性更好）
            img.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 返回纯base64字符串（关键修正：去掉data:...前缀）
            return img_base64
    except Exception as e:
        print(f"图片处理错误: {e}")
        return None


def identify_animal(image_path):
    img_base64 = image_to_base64(image_path)
    if not img_base64:
        return "图片处理失败，请检查路径或格式"

    # 调用Ollama API（严格按照官方格式）
    try:
        response = requests.post(
            url="http://10.86.5.43:11434/api/generate",
            json={
                "model": "qwen2.5vl:7b",  # 确保模型名称正确（区分大小写）
                "prompt": "请识别图片中的动物，并告诉我它的名称。",
                "images": [img_base64],  # 纯base64字符串数组（关键修正）
                "temperature": 0.1,
                "stream": False  # 关闭流式响应，直接获取完整结果
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "未识别到动物")
        else:
            # 打印详细错误信息（方便排查）
            print(f"API错误详情: {response.text}")
            return f"API调用失败，状态码: {response.status_code}"
            
    except Exception as e:
        return f"请求发送失败: {e}"


if __name__ == "__main__":
    animal_image_path = "cat.jpg"
    print("图片是否存在：", os.path.exists(animal_image_path))
    
    result = identify_animal(animal_image_path)
    print("识别结果：")
    print(result)