# author: YilEnS e1351599@u.nus.edu
# tools.py - 定义模型调用的工具/结构，强制规范输出格式

import json

TOOLS_CONFIG = {
    "developer": {
        "name": "submit_chapter",
        "description": "提交编写好的小说正文。",
        "properties": {
            "content": {"type": "string",
                        "description": "完整的小说正文内容。只需包含正文，不可包含任何解释、说明或HTML标签。"}
        },
        "required": ["content"]
    },
    "compressor": {
        "name": "submit_summary",
        "description": "提交提取的剧情摘要。",
        "properties": {
            "summary": {"type": "string", "description": "精炼的剧情摘要文本。"}
        },
        "required": ["summary"]
    },
    "designer": {
        "name": "submit_plan",
        "description": "提交章节计划和大纲。",
        "properties": {
            "plan": {"type": "string", "description": "详细的章节计划、大纲和要点。"}
        },
        "required": ["plan"]
    },
    "reviewer": {
        "name": "submit_feedback",
        "description": "提交评审意见或打分反馈。",
        "properties": {
            "feedback": {"type": "string", "description": "详细的评审意见，可以包含打分信息和具体建议。"}
        },
        "required": ["feedback"]
    },
    "judge": {
        "name": "submit_score",
        "description": "提交最终得分。",
        "properties": {
            "score": {"type": "integer", "description": "对文本质量的评分，范围0-100。"}
        },
        "required": ["score"]
    },
    "archiver": {
        "name": "submit_archive",
        "description": "提交人物设定的更新分析。",
        "properties": {
            "status": {"type": "string", "description": "填'无需更新'如果人物设定没变，填'需要更新'如果有变动。"},
            "updated_characters": {"type": "string",
                                   "description": "如果需要更新，这里填写完整的、更新后的人物设定。否则留空。"}
        },
        "required": ["status", "updated_characters"]
    }
}


def get_tool_definition(api_type, tool_config):
    if api_type == "Gemini":
        from google.genai import types
        properties = {}
        for prop_name, prop_info in tool_config["properties"].items():
            prop_type = "INTEGER" if prop_info["type"] == "integer" else "STRING"
            properties[prop_name] = types.Schema(type=prop_type, description=prop_info["description"])

        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=tool_config["required"]
                    )
                )
            ]
        )
    else:
        return {
            "type": "function",
            "function": {
                "name": tool_config["name"],
                "description": tool_config["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_config["properties"],
                    "required": tool_config["required"]
                }
            }
        }


def get_tool_for_role(role_key, api_type):
    if role_key not in TOOLS_CONFIG:
        return None
    return get_tool_definition(api_type, TOOLS_CONFIG[role_key])


def extract_tool_result(response_obj, api_type, role_key):
    config = TOOLS_CONFIG.get(role_key)
    if not config:
        return None

    extracted_args = None

    if api_type == "Gemini":
        try:
            if not response_obj.candidates:
                return None
            parts = response_obj.candidates[0].content.parts
            for part in parts:
                if getattr(part, "function_call", None) and part.function_call.name == config["name"]:
                    args = part.function_call.args
                    if isinstance(args, dict):
                        extracted_args = args
                    elif hasattr(args, "items"):
                        extracted_args = {k: v for k, v in args.items()}
                    else:
                        extracted_args = {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}
                    break
        except Exception:
            pass

    else:
        try:
            message = response_obj.choices[0].message
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    if tool_call.function.name == config["name"]:
                        extracted_args = json.loads(tool_call.function.arguments)
                        break
        except Exception:
            pass

    if extracted_args is not None:
        if len(config["required"]) == 1:
            key = config["required"][0]
            if key in extracted_args:
                return extracted_args[key]
        return extracted_args

    return None