from textwrap import dedent
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from utils import naver_search, completion, highlighter, load_env

query = input("\033[38;5;119m질문을 입력하세요:\033[0m\n")
load_env()

# 1. 키워드 추출
response = completion(query)
keyword = response.choices[0].message.content

# 2. 네이버 백과사전 검색
search_result = naver_search(keyword, category="encyc", display=10)
urls = [item['link'] for item in search_result['items']]
# HTML 파싱
loader = WebBaseLoader(
    web_paths=(urls),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("headword", "txt")
        )
    ),
)
docs = loader.load()

# 3. 전처리
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# 텍스트 10글자 미만은 필터링
splits = list(filter(lambda x: len(x.page_content) >=10, splits))

# 4. store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

# 5. retrieve
retriever = vectorstore.as_retriever(search_tyle="similarity", search_kwargs={'k': 3})

# 6. generate
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(f"\033[38;5;119m질문:\033[0m\n{query}")
print(f"\033[38;5;119m답변:\033[0m")
for chunk in rag_chain.stream(query):
    print(chunk, end='', flush=True)