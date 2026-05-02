# author: YilEnS e1351599@u.nus.edu
# api.py - 模型 API 调用与网络请求层

import json
import urllib.request
import urllib.error
import threading
from google import genai
from google.genai import types
from openai import OpenAI
import tools

# ==========================================
# API 客户端服务层
# ==========================================
class LLMService:
    @staticmethod
    def call_gemini(api_key, model_name, sys_inst, final_prompt, model_intro, pre_history, history_text, temperature,
                    top_p, top_k, role_key=None):
        client = genai.Client(api_key=api_key)
        safeties = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in
                    ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                     "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        config_kwargs = {
            "system_instruction": sys_inst,
            "safety_settings": safeties,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # 通过 tools.py 获取规范化 JSON Schema 工具
        if role_key:
            tool = tools.get_tool_for_role(role_key, "Gemini")
            if tool:
                config_kwargs["tools"] = [tool]
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",  # 强制调用预设好的纯净提取方法
                        allowed_function_names=[tools.TOOLS_CONFIG[role_key]["name"]]
                    )
                )

        config = types.GenerateContentConfig(**config_kwargs)

        contents = [types.Content(role="user", parts=[types.Part.from_text(text="自我介绍一下。")]),
                    types.Content(role="model", parts=[types.Part.from_text(text=model_intro)])]
        for turn in (pre_history or []):
            contents.extend([types.Content(role="user", parts=[types.Part.from_text(text=turn["user"])]),
                             types.Content(role="model", parts=[types.Part.from_text(text=turn["model"])])])
        if history_text and history_text.strip():
            contents.extend([types.Content(role="user", parts=[types.Part.from_text(text="回顾一下之前的剧情。")]),
                             types.Content(role="model", parts=[
                                 types.Part.from_text(text=f"下面是我之前已经完成的剧情：\n{history_text}")])])
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)]))

        try:
            res = client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as e:
            # 万一底层模型不支持Tool回退到纯文本模式
            if "tool" in str(e).lower() or "function" in str(e).lower():
                config_kwargs.pop("tools", None)
                config_kwargs.pop("tool_config", None)
                config_fallback = types.GenerateContentConfig(**config_kwargs)
                res = client.models.generate_content(model=model_name, contents=contents, config=config_fallback)
            else:
                raise e

        if not res.candidates: raise ValueError("API未返回候选结果")

        # 拦截Tool解析成规范结构
        if role_key:
            result = tools.extract_tool_result(res, "Gemini", role_key)
            if result is not None:
                return result

        if res.candidates[0].finish_reason and "STOP" not in str(res.candidates[0].finish_reason):
            raise ValueError(f"生成异常终止: {res.candidates[0].finish_reason}")
        return res.text

    @staticmethod
    def call_openai(api_key, api_url, model_name, sys_inst, final_prompt, model_intro, pre_history, history_text,
                    temperature, top_p, role_key=None):
        client = OpenAI(api_key=api_key, **({"base_url": api_url} if api_url else {}))
        msgs = [{"role": "system", "content": sys_inst}, {"role": "user", "content": "自我介绍一下。"},
                {"role": "assistant", "content": model_intro}]
        for t in (pre_history or []):
            msgs.extend([{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["model"]}])
        if history_text and history_text.strip():
            msgs.extend([{"role": "user", "content": "之前写了哪些剧情？"},
                         {"role": "assistant", "content": f"以下是我之前已经完成的剧情：\n{history_text}"}])
        msgs.append({"role": "user", "content": final_prompt})

        kwargs = {
            "model": model_name,
            "messages": msgs,
            "temperature": temperature,
            "top_p": top_p
        }

        if role_key:
            tool = tools.get_tool_for_role(role_key, "OpenAI")
            if tool:
                kwargs["tools"] = [tool]
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tools.TOOLS_CONFIG[role_key]["name"]}}

        try:
            res = client.chat.completions.create(**kwargs)
        except Exception as e:
            # 若自建 API 不支持 Function Calling 会报错，在此做稳健回退
            if "tool" in str(e).lower() or "function" in str(e).lower() or "schema" in str(e).lower():
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                res = client.chat.completions.create(**kwargs)
            else:
                raise e

        if role_key:
            result = tools.extract_tool_result(res, "OpenAI", role_key)
            if result is not None:
                return result

        return res.choices[0].message.content


class APIKeyManager:
    def __init__(self):
        self.indices, self.lock = {}, threading.Lock()

    def get_next_key(self, keys_str):
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys: raise ValueError("未配置 API Keys")
        with self.lock:
            idx = self.indices.get(keys_str, 0)
            self.indices[keys_str] = idx + 1
            return keys[idx % len(keys)]


def fetch_models(api_type, url_str, keys_str):
    if not keys_str: return [], "❌ 请先填入 API Key"
    api_key = keys_str.split(',')[0].strip()
    try:
        if api_type == "Gemini":
            client = genai.Client(api_key=api_key)
            models = [m.name.replace('models/', '') for m in client.models.list()]
        else:
            base_url = url_str.rstrip('/')
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{base_url}/models", headers=headers),
                                            timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                if e.code == 404 and not base_url.endswith('/v1'):
                    with urllib.request.urlopen(urllib.request.Request(f"{base_url}/v1/models", headers=headers),
                                                timeout=10) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                else:
                    raise e
            models = [m['id'] for m in data.get('data', []) if 'id' in m]
        if not models: return [], "⚠️ 返回空模型列表"
        return models, f"✅ 获取 {len(models)} 个可用模型"
    except Exception as e:
        return [], f"❌ 获取失败:\n{e}"