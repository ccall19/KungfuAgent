from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import ZhipuAIEmbeddings, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import yaml
from langchain_core.messages import SystemMessage, trim_messages
from fastapi.responses import StreamingResponse

config = yaml.safe_load(open("configs/config-sword.yaml", "r", encoding="utf-8"))

# 文档加载与分割
loader = PyPDFLoader(config['file_path'])
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, add_start_index=True)
split_docs = text_splitter.split_documents(pages)

# 向量库
vector_store = InMemoryVectorStore.from_documents(split_docs, ZhipuAIEmbeddings(api_key=config['api']['embedding_key']))
retriever = vector_store.as_retriever()

model_name = config.get("model_name", 'model_zhipu')
model_config = config.get(model_name, {})
# LLM与Prompt
llm = ChatOpenAI(
    model=model_config.get('model', 'glm-4-plus'),
    openai_api_key=model_config.get('api_key'),
    base_url=model_config.get('base_url', 'https://open.bigmodel.cn/api/paas/v4/'),
    temperature=model_config.get('temperature', 0.7),
)

trimmer_config = config.get("trimmer", {})
trimmer = trim_messages(
    max_tokens=trimmer_config.get("max_tokens", 50000),
    strategy=trimmer_config.get("strategy", "last"),
    token_counter=llm,
    include_system=trimmer_config.get("include_system", True),
    allow_partial=trimmer_config.get("allow_partial", False),
    start_on=trimmer_config.get("start_on", "human"),
)

system_prompt = config.get("system_prompt")


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"],
)

class Query(BaseModel):
    question: str
    session_id: str
class Request(BaseModel):
    session_id: str

@app.get("/history")
async def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = history.messages
    return messages

@app.post("/ask")
async def ask(query: Query):
    async def event_stream():
        async for chunk in conversational_rag_chain.astream(
            {"input": query.question},
            config={"configurable": {"session_id": query.session_id}}
        ):
            if chunk.get("answer"):
                yield f"data: {chunk['answer']}\n\n"  # SSE协议格式
        yield "data: [DONE]\n\n"  # 结束信号

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)