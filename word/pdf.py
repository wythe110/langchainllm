from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

# 配置Ollama连接
OLLAMA_HOST = "http://10.86.5.43:11434"

def process_pdf(file_path: str, query: str):
    """
    处理PDF文档并回答用户问题
    
    参数:
    file_path: PDF文档路径
    query: 用户提出的问题
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    # 2. 加载PDF文档
    print(f"正在加载PDF文档: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 3. 文档分块处理
    print("正在分割文档内容...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每块1000字符
        chunk_overlap=200,  # 块间重叠200字符
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档分割为 {len(chunks)} 个文本块")
    
    # 4. 创建向量存储
    print("正在生成文本嵌入...")
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model="qwen3:14b"  # 注意使用正确的模型名称
    )
    
    # 使用Chroma向量数据库（避免FAISS安装问题）
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"  # 本地存储路径
    )
    
    # 5. 初始化LLM
    print("正在初始化语言模型...")
    llm = Ollama(
        base_url=OLLAMA_HOST,
        model="qwen3:14b",
        temperature=0.3,  # 适度创造性
        top_k=50,         # 增加多样性
        top_p=0.9         # 核采样
    )
    
    # 6. 创建问答链
    print("构建问答系统...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 简单拼接上下文
        retriever=vector_store.as_retriever(
            search_type="mmr",  # 最大边际相关性搜索
            search_kwargs={"k": 5}  # 检索前5个相关片段
        ),
        return_source_documents=True,
        verbose=True  # 显示详细处理过程
    )
    
    # 7. 执行查询
    print(f"正在处理问题: {query}")
    result = qa_chain.invoke({"query": query})
    
    # 8. 打印结果
    print("\n" + "="*50)
    print(f"问题: {query}")
    print("\n答案:")
    print(result['result'])
    
    # 9. 显示来源片段
    print("\n来源文档片段:")
    for i, doc in enumerate(result['source_documents'][:3]):  # 显示前3个来源
        print(f"\n[片段 {i+1} - 页码: {doc.metadata['page']+1}]")
        print(doc.page_content[:300] + "...")  # 显示前300字符

if __name__ == "__main__":
    # 示例使用 - 替换为你的PDF路径和问题
    pdf_path = r"C:\Users\jinpeng.kuang\Desktop\ocam.pdf"  # 你的PDF路径
    user_query = "简略讲一下这篇论文讲了什么呢？"  # 你的问题
process_pdf(pdf_path, user_query)