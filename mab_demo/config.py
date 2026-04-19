"""全局配置: 所有环境变量集中在此, 方便生产环境覆盖。

支持的环境变量:
  MAB_LLM_API_KEY        OpenAI 兼容 API key (设空则所有 LLM 调用 fallback 到 mock)
  MAB_LLM_BASE_URL       OpenAI 兼容 endpoint (默认 DeepSeek)
  MAB_LLM_MODEL          模型名 (默认 deepseek-chat)
  MAB_LLM_TIMEOUT        请求超时(秒)
  MAB_LLM_TEMPERATURE    全局默认温度
  MAB_LLM_RETRIES        失败重试次数
  MAB_RM_TYPE            奖励模型类型: rule_based | llm_judge | external_rm
  MAB_RM_ENDPOINT        外部奖励模型 endpoint (rm_type=external_rm 时生效)
  MAB_USE_CACHE          是否启用先验缓存 (0/1)
  MAB_CAUSAL_USE_LLM     因果去偏是否启用 LLM (0/1, 默认 0 只用规则)
  MAB_FALLBACK_ON_FAIL   LLM 失败时是否回退到 mock (0/1, 默认 1)
"""
import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass
class Config:
    # LLM endpoint
    api_key: str = os.getenv("MAB_LLM_API_KEY", "")
    base_url: str = os.getenv("MAB_LLM_BASE_URL", "https://api.deepseek.com/v1")
    model: str = os.getenv("MAB_LLM_MODEL", "deepseek-chat")
    timeout: int = _env_int("MAB_LLM_TIMEOUT", 30)
    default_temperature: float = _env_float("MAB_LLM_TEMPERATURE", 0.2)
    retries: int = _env_int("MAB_LLM_RETRIES", 2)

    # Reward Model
    rm_type: str = os.getenv("MAB_RM_TYPE", "rule_based")
    rm_endpoint: str = os.getenv("MAB_RM_ENDPOINT", "")

    # Flags
    use_cache: bool = _env_bool("MAB_USE_CACHE", True)
    causal_use_llm: bool = _env_bool("MAB_CAUSAL_USE_LLM", False)
    fallback_on_fail: bool = _env_bool("MAB_FALLBACK_ON_FAIL", True)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)


CONFIG = Config()
