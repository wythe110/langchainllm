from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os

# 配置Ollama连接
OLLAMA_HOST = "http://10.86.5.43:11434"

#def process_document(file_path: r"C:\Users\jinpeng.kuang\Desktop\cy.docx", query: "帮我写一个关于周末出游的日记"):
def process_document(file_path: str, query: str = "这篇日记的作者早上8点做了什么"):
    """
    #处理Word文档并回答用户问题
    
    参数:
    file_path: Word文档路径 (.docx)
    query: 用户提出的问题
    """
    # 1. 加载文档
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    
    # 2. 文档分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每块1000字符
        chunk_overlap=200,  # 块间重叠200字符
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. 创建向量存储
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model="qwen3:14b" #这里是qwen3：14b还是qwen：14b？
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. 初始化LLM
    llm = Ollama(
        base_url=OLLAMA_HOST,
        model="qwen3:14b",#模型是什么？
        temperature=0.2
    )
    
    # 5. 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    # 6. 执行查询
    result = qa_chain.invoke({"query": query})
    # 7. 打印结果
    print(f"问题: {query}")
    print(f"答案: {result['result']}")
    print("\n来源文档片段:")
    for i, doc in enumerate(result['source_documents'][:2]):  # 显示前2个来源
        print(f"[片段 {i+1}]: {doc.page_content[:200]}...")

if __name__ == "__main__":
    # 示例使用
    doc_path = r"C:\Users\jinpeng.kuang\Desktop\cy.docx"  # 替换为你的文档路径
    user_query = "这篇日记的作者早上8点做了什么"
    
    if not os.path.exists(doc_path):
        print(f"错误: 文件 {doc_path} 不存在")
    else:
        process_document(doc_path, user_query)