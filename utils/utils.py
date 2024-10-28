import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Callable, Dict
import json
import os
import time
import random

from pygments import highlight, lexers, formatters, styles
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

def find_file(fname: str, start_path: str = None):
    """fname으로 된 파일을 star_path가 None일 경우, apt_trade/ 부터 재귀적으로 검색합니다.
    가장 최근에 검색된 fname 1개만을 리턴합니다.

    Args:
        fname: 찾을 파일명(확장자 포함)
        start_path: 검색을 시작할 최상위 폴더 트리

    Returns: 검색된 파일의 abspath

    """
    if not start_path:
        start_path = os.getcwd()
    if not os.path.exists(start_path):
        raise ValueError(f"{start_path} does not exists.")

    paths = []
    for current_path, _, file_list in os.walk(start_path):
        if fname in file_list:
            paths.append(os.path.join(current_path, fname))
    if not paths:
        raise FileExistsError(f"'{fname}' file doesn't exists.")
    if len(paths) == 1:
        return paths[0]
    if len(paths) > 1:
        return paths

def load_env(key: str = None, fname=".env", start_path=None):
    """1) 환경변수 설정이 되었는지 검색하고, 2) 설정값이 없으면 환경변수가 정의된 파일을찾는다
    Args:
        key: 검색할 환경변수 키
        fname: 환경변수를 설정한 파일명, default ".env"
        start_path: fname을 찾을 최상위 폴더, 해당 root(apt_trade)에서부터 sub folder를 재귀적으로 탐색

    """

    env_path = find_file(fname, start_path=start_path)
    if isinstance(env_path, list) and len(env_path) > 1:
        raise ValueError(f"{env_path=}\nmultiple files detected. please make unique file")
    load_dotenv(env_path)
    if key:
        env = os.getenv(key, None)
        if not env:
            raise ValueError(f"cant find env variable '{key}'")
        return env

def highlighter(obj: Union[List, str, Dict], lexer: str = None, formatter: str = None, style: str = None):
    if not lexer:
        lexer = "json"
    if lexer == "json":
        obj = json.dumps(obj, indent=4, ensure_ascii=False)
    lexer = lexers.find_lexer_class_by_name(lexer)

    if not formatter:
        formatter = "terminal256"
    formatter = formatters.find_formatter_class(formatter)

    if not style:
        style = "lightbulb"

    encoded = highlight(obj, lexer=lexer(), formatter=formatter(style=style))
    print(encoded)


def concurrent_execute(func: Callable, iterables: List, n_workers: int = None, safe_wait: bool = True):
    if not n_workers:
        n_workers = os.cpu_count() - 3

    logger.info(f"n_workers: {n_workers}")
    logger.info(f"iterables length: {len(iterables)}")

    futures = []
    result = []
    errors = []
    # threadpool 정의
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for iters in iterables:
            future = executor.submit(func, iters)
            future.task_id = iters # 예외처리를 위해 task_id 할당
            futures.append(future)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if safe_wait:
                time.sleep(random.randint(1, 10))
            try:
                result.append(future.result(timeout=600))
            except Exception as e:
                logger.error(f"{repr(e)}\nError occurs in '{future.task_id}'. Check return value of errors")
                errors.append(future.task_id)
    return result, errors
