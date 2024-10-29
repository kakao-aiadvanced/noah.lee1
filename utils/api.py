from openai import OpenAI
from typing import Literal
from textwrap import dedent
import json
import requests
from pathlib import Path
from .utils import load_env

def naver_search(
        query: str,
        api_client_id: str = None,
        api_client_secret: str = None,
        category: Literal["blog", "news", "kin", "encyc", "cafearticle", "webkr"] = "news",
        display: int = 20,
        start: int = 1,
        sort: str = "sim"
):
    """ Naver 검색 API를 통해 데이터를 수집합니다.
    수집한 데이터는 모델이 generate할 때 RAG로 사용할 수 있습니다.
    자세한 내용은 https://developers.naver.com/docs/serviceapi/search/blog/blog.md를 참고하세요.

    Args:
        query: 검색하려는 쿼리값
        api_client_id: 네이버 검색 API이용을 위한 발급 받은 client_id 값
          - 환경변수는 'NAVER_API_ID'로 설정하세요
        api_client_secret: 네이버 검색 API이용을 위한 발급 받은 client_secret 값
          - 환경변수는 'NAVER_API_SECRET'으로 설정하세요
        category: 검색하려는 카테고리, 아래 카테고리로 검색이 가능합니다
          - blog: 블로그
          - news: 뉴스
          - kin: 지식인
          - encyc: 백과사전
          - cafearticle: 카페 게시글
          - webkr: 웹문서
        display: 검색 결과 수 지정, default = 20
        start: 검색 페이지 값
        sort: 정렬값
          - 'sim': 정확도 순으로 내림차순 정렬
          - 'date': 날짜 순으로 내림차순 정렬

    Returns: API로부터 제공받은 검색 결과 response값

    """

    if not (api_client_id and api_client_secret):
        api_client_id = load_env("NAVER_API_ID", start_path=str(Path(__file__).parent.parent))
        api_client_secret = load_env("NAVER_API_SECRET", start_path=str(Path(__file__).parent.parent))
    if not api_client_id or not api_client_secret:
        id_ok = "'NAVER_API_ID'" if not api_client_id else ""
        secret_ok = "'NAVER_API_SECRET'" if not api_client_id else ""
        raise ValueError(f"{id_ok} {secret_ok} Not setted")

    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": api_client_id, "X-Naver-Client-Secret": api_client_secret}
    query = query.encode("utf8")
    params = {"query": query, "start": start, "display": display, "sort": sort}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode())
    else:
        return response.raise_for_status()


def completion(user_prompt: str, system_prompt: str = None, model: str = "gpt-4o-mini"):
    client = OpenAI()
    if not system_prompt:
        system_prompt = dedent("""\
        사용자가 제공하는 질문에 대해 핵심 키워드를 추출하세요.\
        질문을 가장 대표할 수 있는 핵심 키워드 1개만 추출하세요.
        아래 예시를 참고하세요
        예시1)
        ```
        ### 질문: 충당부채는 무엇이며, 왜 기업 재무제표에서 중요한 항목인가요?
        충당부채
        ```
        예시2)
        ### 질문: 폐렴의 초기 증상은 무엇이며, 감기나 독감과 어떻게 구분할 수 있나요?
        폐렴
        ```
        """)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages
    )
    return response