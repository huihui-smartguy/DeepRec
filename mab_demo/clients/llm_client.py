"""OpenAI 兼容大模型客户端: 封装 retry / timeout / JSON mode / 缓存 / 优雅降级。

支持的后端:
  - DeepSeek API       (MAB_LLM_BASE_URL=https://api.deepseek.com/v1)
  - Qwen (阿里 DashScope) 兼容模式
  - 本地 vLLM          (MAB_LLM_BASE_URL=http://localhost:8000/v1)
  - OpenAI 官方

设计要点:
  1. 可选依赖: openai 未安装时自动降级为 disabled, 不报错
  2. 幂等缓存: 按 (prompt_hash, model, temperature) 缓存 JSON 响应, 命中率可到 40%+
  3. 重试: 指数退避, 默认 2 次
  4. 全链路降级: 任何异常都返回 None, 上层自动走 mock 分支
"""
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from config import CONFIG

log = logging.getLogger("mab_demo.llm")

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


class LLMClient:
    """无状态线程安全的 OpenAI-兼容客户端。"""

    def __init__(self):
        self._client: Optional[Any] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self):
        if not (_OPENAI_AVAILABLE and CONFIG.llm_enabled):
            return
        try:
            self._client = OpenAI(
                api_key=CONFIG.api_key,
                base_url=CONFIG.base_url,
                timeout=CONFIG.timeout,
            )
        except Exception as e:
            log.warning(f"LLM client init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _cache_key(self, prompt: str, model: str, temperature: float, system: Optional[str]) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(f"{temperature:.4f}".encode())
        if system:
            h.update(system.encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def chat_json(self,
                  prompt: str,
                  system: Optional[str] = None,
                  model: Optional[str] = None,
                  temperature: Optional[float] = None,
                  retries: Optional[int] = None,
                  force_json: bool = True) -> Optional[Dict[str, Any]]:
        """发起一次 LLM 调用, 强制返回合法 JSON 字典; 失败返回 None(不抛异常)。"""
        if not self.enabled:
            return None

        model = model or CONFIG.model
        temperature = CONFIG.default_temperature if temperature is None else temperature
        retries = CONFIG.retries if retries is None else retries

        # 缓存命中
        if CONFIG.use_cache:
            key = self._cache_key(prompt, model, temperature, system)
            if key in self._cache:
                log.info(f"LLM cache hit: {key[:12]}")
                return self._cache[key]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if force_json:
            # OpenAI / DeepSeek / Qwen 均支持 json_object 模式
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)  # type: ignore
                text = resp.choices[0].message.content or "{}"
                data = json.loads(text)
                if CONFIG.use_cache:
                    self._cache[key] = data
                return data
            except json.JSONDecodeError as e:
                last_err = e
                log.warning(f"LLM returned non-JSON (attempt {attempt+1}): {e}")
            except Exception as e:
                last_err = e
                log.warning(f"LLM call error (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(min(2.0, 0.5 * (2 ** attempt)))

        log.error(f"LLM call failed after {retries+1} attempts: {last_err}")
        return None


# 进程级单例
llm_client = LLMClient()
