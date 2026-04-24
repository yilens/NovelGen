# author: YilEnS e1351599@u.nus.edu
# backend.py - 后端核心逻辑层

import json, os, re, urllib.request, urllib.error, time, random, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from openai import OpenAI
import mode  # 假设存在于您的环境中

# ==========================================
# 常量配置与全局初始化
# ==========================================
MAX_API_RETRIES = 10

ROLE_MAP = {
    "设计者": "designer", "开发者": "developer", "评审者": "reviewer",
    "裁判者": "judge", "压缩者": "compressor", "清洗者": "cleaner", "归档者": "archiver"
}

AGENT_NAMES_MAP = [
    ("设计者", "designer", "Designer_R18.json"),
    ("开发者", "developer", "Developer_R18.json"),
    ("评审者", "reviewer", "Reviewer_R18.json"),
    ("裁判者", "judge", "Judger_R18.json"),
    ("压缩者", "compressor", "Compressor_R18.json"),
    ("清洗者", "cleaner", "Cleaner_R18.json"),
    ("归档者", "archiver", "Archiver_R18.json")
]

BASE_CONFIG_KEYS = [
    ("book_title", ""), ("outline", ""), ("style", ""), ("characters", ""),
    ("use_manual_outline", False), ("manual_outline_data", [[1, "", ""]]),
    ("global_prompt", ""), ("custom_style_prompt", ""),
    ("use_ai_reviewer", True), ("need_dev_revise", False), ("use_ai_cleaner", False), ("use_archiver", False),
    ("context_max_chars", 50000), ("target_chapter", 0),
    ("api_type", "Gemini"), ("api_keys", ""), ("api_url", ""), ("api_model", "gemini-2.5-flash"),
    ("fallback_api_type", "OpenAI Compatible"), ("fallback_api_keys", ""), ("fallback_api_url", ""),
    ("fallback_api_model", "")
]

AGENT_CONFIG_KEYS = []
for _, en, d_val in AGENT_NAMES_MAP:
    use_hist_default = True if en in ["designer", "developer"] else False
    ctx_count_default = 20 if en == "developer" else 0

    AGENT_CONFIG_KEYS.extend([
        (f"{en}_mode", d_val), (f"{en}_prompt", ""), (f"{en}_count", 1),
        (f"{en}_use_history", use_hist_default), (f"{en}_full_ctx_count", ctx_count_default),
        (f"{en}_temperature", 0.9 if en != "designer" else 0.7),
        (f"{en}_top_p", 0.9), (f"{en}_top_k", 40),
        (f"{en}_api_type", "Gemini"), (f"{en}_api_keys", ""), (f"{en}_api_url", ""),
        (f"{en}_api_model", "gemini-2.5-flash")
    ])

ALL_CONFIG_KEYS = BASE_CONFIG_KEYS + AGENT_CONFIG_KEYS


# ==========================================
# 通用文件与路径操作工具
# ==========================================
def load_json(filepath, default=None):
    if default is None: default = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def write_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def get_dir(*subdirs):
    return os.path.join("NovelGen", *subdirs)


def get_all_modes():
    mode_dir = get_dir("modes")
    return sorted([f for f in os.listdir(mode_dir) if f.endswith('.json')]) if os.path.exists(mode_dir) else []


def safe_name(name):
    return re.sub(r'[\\/:*?"<>|.]', '_', name.strip())


def build_config_dict(*args):
    return dict(zip([k[0] for k in ALL_CONFIG_KEYS], args))


def save_config_json(config_dict):
    save_json(get_dir("user_input.json"), config_dict)
    if bt := config_dict.get("book_title", "").strip():
        save_json(get_dir("Books", safe_name(bt), "book_config.json"), config_dict)
    return "✅ 配置已成功保存！"


# ==========================================
# 数据与文件管理模块
# ==========================================
class ModeManager:
    @staticmethod
    def save_mode(mode_name, sys_prompt, intro, history_data):
        if not mode_name.strip():
            return "❌ 配置名不能为空", []
        name = mode_name.strip()
        name += "" if name.endswith(".json") else ".json"

        history = [{"user": str(r[0]).strip(), "model": str(r[1]).strip()} for r in history_data if
                   len(r) >= 2 and (str(r[0]).strip() or str(r[1]).strip())]
        save_json(get_dir("modes", name),
                  {"system_prompt": sys_prompt.strip(), "intro": intro.strip(), "history": history})

        return f"✅ 预设 [{name}] 保存成功！", get_all_modes()


class VolumeManager:
    @staticmethod
    def get_chapter_dir(book_title):
        return get_dir("Books", safe_name(book_title), "chapters")

    @staticmethod
    def get_summary_dir(book_title):
        return get_dir("Books", safe_name(book_title), "summaries")

    @staticmethod
    def ensure_directories_and_migrate(book_title):
        ch_dir = VolumeManager.get_chapter_dir(book_title)
        sum_dir = VolumeManager.get_summary_dir(book_title)
        os.makedirs(ch_dir, exist_ok=True)
        os.makedirs(sum_dir, exist_ok=True)

        if not os.listdir(ch_dir):
            full_file = get_dir("Books", safe_name(book_title), f"{safe_name(book_title)}_full_volume.txt")
            if os.path.exists(full_file):
                text = read_file(full_file)
                for m in re.finditer(r"第(\d+)章\n(.*?)(?=\n\n第\d+章\n|$)", text, re.DOTALL):
                    write_file(os.path.join(ch_dir, f"{m.group(1)}.txt"), m.group(2).strip())

        if not os.listdir(sum_dir):
            comp_file = get_dir("Books", safe_name(book_title), f"{safe_name(book_title)}_compressed_volume.txt")
            if os.path.exists(comp_file):
                text = read_file(comp_file)
                for m in re.finditer(r"第(\d+)章剧情：(.*?)(?=\n第\d+章剧情：|$)", text, re.DOTALL):
                    write_file(os.path.join(sum_dir, f"{m.group(1)}.txt"), m.group(2).strip())

    @staticmethod
    def rebuild_full(book_title):
        d = VolumeManager.get_chapter_dir(book_title)
        full_file = get_dir("Books", safe_name(book_title), f"{safe_name(book_title)}_full_volume.txt")
        content = []
        if os.path.exists(d):
            files = sorted([f for f in os.listdir(d) if f.endswith('.txt') and f[:-4].isdigit()],
                           key=lambda x: int(x[:-4]))
            for f in files:
                content.append(f"第{f[:-4]}章\n{read_file(os.path.join(d, f))}")

        final_text = "\n\n".join(content)
        write_file(full_file, final_text)
        return final_text

    @staticmethod
    def rebuild_compressed(book_title):
        d = VolumeManager.get_summary_dir(book_title)
        comp_file = get_dir("Books", safe_name(book_title), f"{safe_name(book_title)}_compressed_volume.txt")
        content = []
        if os.path.exists(d):
            files = sorted([f for f in os.listdir(d) if f.endswith('.txt') and f[:-4].isdigit()],
                           key=lambda x: int(x[:-4]))
            for f in files:
                content.append(f"第{f[:-4]}章剧情：{read_file(os.path.join(d, f))}")

        final_text = "\n".join(content)
        write_file(comp_file, final_text)
        return final_text


class FileManager:
    @staticmethod
    def get_books():
        user_dir = get_dir("Books")
        if not os.path.exists(user_dir): return []
        return [d for d in os.listdir(user_dir) if
                os.path.isdir(os.path.join(user_dir, d)) and d not in ["mode", "modes", "outline", "roles"]]

    @staticmethod
    def export_book_folder(raw_title):
        if not raw_title: return None, "❌ 缺少书名信息"
        book_title, book_dir = safe_name(raw_title), get_dir("Books", safe_name(raw_title))
        if not os.path.exists(book_dir): return None, f"❌ 找不到对应文件夹"

        zip_base_path = get_dir("Books", f"{book_title}_完整包")
        shutil.make_archive(zip_base_path, 'zip', book_dir)
        return f"{zip_base_path}.zip", f"✅ 已打包 {book_title}，请点击下载。"

    @staticmethod
    def delete_book(raw_title):
        if not raw_title: return "❌ 缺少书名信息", []
        book_title = safe_name(raw_title)
        book_dir = get_dir("Books", book_title)
        if os.path.exists(book_dir):
            shutil.rmtree(book_dir)
            zip_file = get_dir("Books", f"{book_title}_完整包.zip")
            if os.path.exists(zip_file): os.remove(zip_file)
            return f"✅ 成功删除小说: {book_title}", FileManager.get_books()
        return "❌ 找不到指定小说的文件夹", FileManager.get_books()

    @staticmethod
    def _get_list(subdir):
        d = get_dir(subdir)
        return sorted([f for f in os.listdir(d) if f.endswith('.json')]) if os.path.exists(d) else []

    @staticmethod
    def _save_json_ui(subdir, name, data, msg):
        if not name.strip() or not data: return "❌ 数据无效", FileManager._get_list(subdir)
        s_name = safe_name(name) + (".json" if not safe_name(name).endswith('.json') else "")
        try:
            save_json(get_dir(subdir, s_name), data)
            return f"✅ {msg} [{s_name}] 保存成功！", FileManager._get_list(subdir)
        except Exception as e:
            return f"❌ 保存失败: {e}", FileManager._get_list(subdir)

    @staticmethod
    def _load_json_ui(subdir, filename, ext_key=None):
        if not filename: return None, "❌ 未选择文件"
        fp = get_dir(subdir, filename)
        if not os.path.exists(fp): return None, f"❌ 找不到: {filename}"
        try:
            data = load_json(fp)
            return data.get(ext_key, "") if ext_key else data, f"✅ 成功加载: {filename}"
        except Exception as e:
            return None, f"❌ 读取失败: {e}"


class ImportManager:
    @staticmethod
    def parse_chapter_number(title_str, fallback_idx):
        """解析字符串中的章节号，支持阿拉伯数字和中文数字"""
        # 匹配阿拉伯数字
        m = re.search(r'第(\d+)[章节回]', title_str)
        if m:
            return int(m.group(1))

        # 匹配中文数字
        m = re.search(r'第([零一二三四五六七八九十百千万〇]+)[章节回]', title_str)
        if m:
            cn_num = m.group(1)
            cn2arb = {'零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
            result, temp = 0, 0
            for char in cn_num:
                if char in cn2arb:
                    temp = cn2arb[char]
                elif char == '十':
                    result += (temp if temp > 0 else 1) * 10
                    temp = 0
                elif char == '百':
                    result += (temp if temp > 0 else 1) * 100
                    temp = 0
                elif char == '千':
                    result += (temp if temp > 0 else 1) * 1000
                    temp = 0
                elif char == '万':
                    result += (temp if temp > 0 else 1) * 10000
                    temp = 0
            result += temp
            if result > 0:
                return result
        return fallback_idx

    @staticmethod
    def import_novel(file_path, raw_title, protagonist_name, config):
        book_title = safe_name(raw_title)
        book_dir = get_dir("Books", book_title)
        os.makedirs(book_dir, exist_ok=True)

        VolumeManager.ensure_directories_and_migrate(book_title)
        ch_dir = VolumeManager.get_chapter_dir(book_title)
        sum_dir = VolumeManager.get_summary_dir(book_title)

        try:
            text = read_file(file_path)
        except Exception as e:
            yield f"❌ 读取文件失败: {e}"
            return

        yield f"✅ 文件读取成功，正在智能检测并切片..."

        pattern = re.compile(
            r"^\s*(?:第[0-9一二三四五六七八九十百千万零〇]+卷\s*)?(第[0-9一二三四五六七八九十百千万零〇]+[章节回].*?)$",
            re.MULTILINE)
        parts = pattern.split(text)

        chapters = []
        if len(parts) == 1:
            chapters.append(parts[0].strip())
        else:
            for i in range(1, len(parts), 2):
                ch_title = parts[i].strip()
                ch_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                ch_title_clean = re.sub(r"^第[0-9一二三四五六七八九十百千万零〇]+[章节回]\s*", "", ch_title).strip()
                if ch_title_clean:
                    ch_content = f"【{ch_title_clean}】\n{ch_content}"
                chapters.append(ch_content)

        if len(parts) > 1 and parts[0].strip():
            if not chapters:
                chapters.append(parts[0].strip())
            else:
                chapters[0] = parts[0].strip() + "\n\n" + chapters[0]

        total_chapters = len(chapters)
        if total_chapters == 0:
            yield "❌ 提取到的章节数为0，请检查文件编码或内容。"
            return

        yield f"✅ 成功提取并切分为 {total_chapters} 章，正在本地强制按序编号并保存..."

        for idx, content in enumerate(chapters):
            write_file(os.path.join(ch_dir, f"{idx + 1}.txt"), content)

        VolumeManager.rebuild_full(book_title)
        save_json(os.path.join(book_dir, f"{book_title}_state.json"), {"completed_chapters": total_chapters})

        yield f"✅ 章节拆分保存完成！准备调用压缩者补充摘要（共 {total_chapters} 章，耗时较长，请耐心等待）..."

        logs = []
        workflow = AgentWorkflow(config, lambda msg: logs.append(msg))
        workflow.is_running = True

        for idx, content in enumerate(chapters):
            ch_num = idx + 1
            workflow.current_chapter = ch_num
            workflow.chapter_texts[ch_num] = content

            yield f"🔄 正在智能压缩第 {ch_num}/{total_chapters} 章...\n{logs[-1] if logs else ''}"
            try:
                summary = workflow.compress_text(content, pass_history=True, protagonist_name=protagonist_name)
                if not summary: summary = "【警告】AI 压缩失败或返回为空，请后续手动修正。"
            except Exception as e:
                summary = f"【错误】压缩过程异常: {e}"

            workflow.summary_texts[ch_num] = summary
            write_file(os.path.join(sum_dir, f"{ch_num}.txt"), summary)

        VolumeManager.rebuild_compressed(book_title)
        yield f"🎉 导入全部处理完成！小说已完全切片并由AI接管进度（共处理 {total_chapters} 章）。"

    @staticmethod
    def import_manual_outline(file_path):
        try:
            text = read_file(file_path)
        except Exception as e:
            return [[1, "", ""]], f"❌ 读取文件失败: {e}"

        pattern = re.compile(
            r"^\s*(?:第[0-9一二三四五六七八九十百千万零〇]+卷\s*)?(第[0-9一二三四五六七八九十百千万零〇]+[章节回].*?)$",
            re.MULTILINE)
        parts = pattern.split(text)

        outline_data = []
        ch_idx_fallback = 1

        for i in range(1, len(parts), 2):
            ch_title_raw = parts[i].strip()
            ch_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            ch_title_clean = re.sub(r"^第[0-9一二三四五六七八九十百千万零〇]+[章节回]\s*", "", ch_title_raw).strip()

            # 智能解析出真实的章节号（如 61）
            actual_ch_num = ImportManager.parse_chapter_number(ch_title_raw, ch_idx_fallback)

            # 填入数据：[章节号(int), 章节名(str), 概要(str)]
            outline_data.append([actual_ch_num, ch_title_clean, ch_content])

            # 为下一章准备一个 fallback 号码（以防下一章没有写“第几章”）
            ch_idx_fallback = actual_ch_num + 1

        if not outline_data:
            if text.strip():
                return [[1, "默认章节", text.strip()]], "⚠️ 未检测到标准章节标题(如“第一章”)，已将全部内容放入第1章。"
            return [[1, "", ""]], "❌ 提取失败：文件为空或格式不匹配。"

        return outline_data, f"✅ 成功导入并切片出 {len(outline_data)} 章大纲！"


class EditManager:
    @staticmethod
    def get_generated_chapters(raw_title):
        if not raw_title: return []
        VolumeManager.ensure_directories_and_migrate(raw_title)
        ch_dir = VolumeManager.get_chapter_dir(raw_title)
        if os.path.exists(ch_dir):
            files = [int(f[:-4]) for f in os.listdir(ch_dir) if f.endswith('.txt') and f[:-4].isdigit()]
            return [f"第{c}章" for c in sorted(files)]
        return []

    @staticmethod
    def load_chapter(raw_title, chapter_str):
        if not raw_title or not chapter_str: return "", "", "❌ 请先选择小说和章节"
        VolumeManager.ensure_directories_and_migrate(raw_title)
        ch_num = re.search(r'\d+', chapter_str).group()
        ch_file = os.path.join(VolumeManager.get_chapter_dir(raw_title), f"{ch_num}.txt")
        sum_file = os.path.join(VolumeManager.get_summary_dir(raw_title), f"{ch_num}.txt")
        return read_file(ch_file), read_file(sum_file), f"✅ 成功加载 {chapter_str}"

    @staticmethod
    def save_chapter(raw_title, chapter_str, new_content, config):
        if not raw_title or not chapter_str: return None, "❌ 缺少必要信息"
        ch_num = re.search(r'\d+', chapter_str).group()

        ch_file = os.path.join(VolumeManager.get_chapter_dir(raw_title), f"{ch_num}.txt")
        write_file(ch_file, new_content.strip())
        VolumeManager.rebuild_full(raw_title)

        try:
            workflow = AgentWorkflow(config, lambda msg: print(msg))
            workflow.is_running = True
            final_sum = workflow.compress_text(new_content)

            if final_sum:
                sum_file = os.path.join(VolumeManager.get_summary_dir(raw_title), f"{ch_num}.txt")
                write_file(sum_file, final_sum)
                VolumeManager.rebuild_compressed(raw_title)

                if app_state.workflow:
                    app_state.workflow.chapter_texts[int(ch_num)] = new_content.strip()
                    app_state.workflow.summary_texts[int(ch_num)] = final_sum
                return final_sum, f"✅ 修改已保存，完整卷已更新，AI 压缩者已重新生成摘要！"
            else:
                return None, f"⚠️ 章节修改成功，但 AI 压缩者异常，请检查配置或重试。"
        except Exception as e:
            return None, f"❌ 修改成功，但后台压缩环节出现错误: {e}"


# ==========================================
# API 客户端服务层
# ==========================================
class LLMService:
    @staticmethod
    def call_gemini(api_key, model_name, sys_inst, final_prompt, model_intro, pre_history, history_text, temperature,
                    top_p, top_k):
        client = genai.Client(api_key=api_key)
        safeties = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in
                    ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                     "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        config = types.GenerateContentConfig(system_instruction=sys_inst, safety_settings=safeties,
                                             temperature=temperature, top_p=top_p, top_k=top_k)

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

        res = client.models.generate_content(model=model_name, contents=contents, config=config)
        if not res.candidates: raise ValueError("API未返回候选结果")
        if res.candidates[0].finish_reason and "STOP" not in str(res.candidates[0].finish_reason):
            raise ValueError(f"生成异常终止: {res.candidates[0].finish_reason}")
        return res.text

    @staticmethod
    def call_openai(api_key, api_url, model_name, sys_inst, final_prompt, model_intro, pre_history, history_text,
                    temperature, top_p):
        client = OpenAI(api_key=api_key, **({"base_url": api_url} if api_url else {}))
        msgs = [{"role": "system", "content": sys_inst}, {"role": "user", "content": "自我介绍一下。"},
                {"role": "assistant", "content": model_intro}]
        for t in (pre_history or []):
            msgs.extend([{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["model"]}])
        if history_text and history_text.strip():
            msgs.extend([{"role": "user", "content": "之前写了哪些剧情？"},
                         {"role": "assistant", "content": f"以下是我之前已经完成的剧情：\n{history_text}"}])
        msgs.append({"role": "user", "content": final_prompt})
        return \
            client.chat.completions.create(model=model_name, messages=msgs, temperature=temperature,
                                           top_p=top_p).choices[
                0].message.content


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


# ==========================================
# 核心业务对象与工作流
# ==========================================
class AgentWorkflow:
    def __init__(self, config, log_callback):
        self.config, self.log = config, log_callback
        self.key_manager = APIKeyManager()
        self.run_event = threading.Event()
        self.run_event.set()

        self.manual_review_event = threading.Event()
        self.manual_review_event.set()

        self.is_running = False

        self.book_title = safe_name(self.config.get("book_title", "未命名小说")) or "未命名小说"
        self.book_dir = get_dir("Books", self.book_title)
        self.err_dir = os.path.join(self.book_dir, "err")

        VolumeManager.ensure_directories_and_migrate(self.book_title)
        self.state_file = os.path.join(self.book_dir, f"{self.book_title}_state.json")
        self.role_file = os.path.join(self.book_dir, f"{self.book_title}_role.json")

        self.full_volume = VolumeManager.rebuild_full(self.book_title)
        self.compressed_volume = VolumeManager.rebuild_compressed(self.book_title)

        state_data = load_json(self.state_file)
        self.current_chapter = state_data.get("completed_chapters", 0) + 1

        self.chapter_texts = {}
        ch_dir = VolumeManager.get_chapter_dir(self.book_title)
        if os.path.exists(ch_dir):
            for f in os.listdir(ch_dir):
                if f.endswith('.txt') and f[:-4].isdigit():
                    self.chapter_texts[int(f[:-4])] = read_file(os.path.join(ch_dir, f))

        self.summary_texts = {}
        sum_dir = VolumeManager.get_summary_dir(self.book_title)
        if os.path.exists(sum_dir):
            for f in os.listdir(sum_dir):
                if f.endswith('.txt') and f[:-4].isdigit():
                    self.summary_texts[int(f[:-4])] = read_file(os.path.join(sum_dir, f))

        self.current_characters = load_json(self.role_file).get("characters", self.config.get("characters", "").strip())
        if not os.path.exists(self.role_file) and self.current_characters:
            save_json(self.role_file, {"characters": self.current_characters})

        self.use_manual, self.manual_dict, self.max_manual_chapter = self.config.get("use_manual_outline", False), {}, 0

        try:
            self.target_chapter = int(self.config.get("target_chapter", 0) or 0)
        except (ValueError, TypeError):
            self.target_chapter = 0

        if self.use_manual:
            for r in self.config.get("manual_outline_data", []):
                if len(r) >= 3 and r[0] and (r[1] or r[2]):
                    self.manual_dict[int(r[0])] = {"name": str(r[1]).strip(), "summary": str(r[2]).strip()}
            if self.manual_dict:
                self.max_manual_chapter = max(self.manual_dict.keys())
                self.log(f"✅ 人工大纲加载成功，目标完结章：第 {self.max_manual_chapter} 章")

                if self.target_chapter <= 0:
                    self.target_chapter = self.max_manual_chapter
                else:
                    self.target_chapter = min(max(self.current_chapter, self.target_chapter), self.max_manual_chapter)
                self.log(f"🎯 设定当前生成目标至：第 {self.target_chapter} 章")
        else:
            if self.target_chapter > 0:
                self.log(f"🎯 设定当前生成目标至：第 {self.target_chapter} 章")

    def _build_dynamic_history(self, full_count):
        actual_full = max(0, min(full_count, self.current_chapter - 1))
        start_full = self.current_chapter - actual_full

        history_parts, summary_parts, full_parts = [], [], []
        for i in range(1, start_full):
            if i in self.summary_texts: summary_parts.append(f"第{i}章剧情：{self.summary_texts[i]}")
        if summary_parts: history_parts.append("【前文剧情摘要】\n" + "\n".join(summary_parts))

        for i in range(start_full, self.current_chapter):
            if i in self.chapter_texts: full_parts.append(f"第{i}章\n{self.chapter_texts[i]}")
        if full_parts: history_parts.append("【前文完整剧情内容】\n" + "\n\n".join(full_parts))

        full_history = "\n\n".join(history_parts)
        max_chars = int(self.config.get("context_max_chars", 50000))
        if len(full_history) > max_chars:
            full_history = full_history[-max_chars:]
            match = re.search(r'[。！？\n]', full_history)
            if match: full_history = full_history[match.end():].strip()

        return full_history

    def save_local_data(self):
        save_json(self.state_file, {"completed_chapters": self.current_chapter})

    def _save_error_log(self, role, conf, sys_inst, is_tool, model_intro, pre_history, final_prompt, history_text,
                        error_msg):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            write_file(os.path.join(self.err_dir, f"err_{timestamp}_{safe_name(role)}.txt"),
                       f"=== 异常记录 ===\n报错信息: {error_msg}\n模型配置: {conf['model_name']} ({conf['api_type']})\n{'-' * 50}\n【系统指令】:\n{sys_inst}\n\n【用户提示词】:\n{final_prompt}\n")
        except Exception as e:
            self.log(f"⚠️ 无法保存错误日志: {e}")

    def _get_api_config(self, use_fallback, role=""):
        if use_fallback:
            return {"model_name": self.config.get("fallback_api_model", "gemini-2.5-flash"),
                    "api_type": self.config.get("fallback_api_type", "Gemini"),
                    "api_url": self.config.get("fallback_api_url", "").strip(),
                    "api_keys_str": self.config.get("fallback_api_keys", ""), "log_prefix": "【备用API】"}

        b_role = role.split()[0].split('(')[0] if role else ""
        a_prefix = ROLE_MAP.get(b_role, "")
        if a_prefix and self.config.get(f"{a_prefix}_api_keys", "").strip():
            return {"model_name": self.config.get(f"{a_prefix}_api_model", "gemini-2.5-flash"),
                    "api_type": self.config.get(f"{a_prefix}_api_type", "Gemini"),
                    "api_url": self.config.get(f"{a_prefix}_api_url", "").strip(),
                    "api_keys_str": self.config.get(f"{a_prefix}_api_keys", ""), "log_prefix": f"【{b_role}专属】"}

        return {"model_name": self.config.get("api_model", "gemini-2.5-flash"),
                "api_type": self.config.get("api_type", "Gemini"),
                "api_url": self.config.get("api_url", "").strip(), "api_keys_str": self.config.get("api_keys", ""),
                "log_prefix": ""}

    def _build_system_instructions(self, role):
        b_role = role.split()[0].split('(')[0]
        is_tool = "清洗者" in role or "压缩者" in role or "归档者" in role
        a_prefix = ROLE_MAP.get(b_role, "")
        mode_file = get_dir("modes", self.config.get(f"{a_prefix}_mode", ""))
        mode_data = load_json(mode_file) if os.path.exists(mode_file) else {}

        sys_tpl = mode_data.get("system_prompt",
                                f"你是机器一样严谨的{b_role}。任务是处理文本，不带情感色彩。" if is_tool else f"你是小说{b_role}，完美执行需求。")
        intro_tpl = mode_data.get("intro",
                                  f"我是{b_role}，可以处理任何类型的文本。" if is_tool else f"您好，我是专属小说{b_role}。")

        sys_inst = sys_tpl.replace("{role}", b_role)
        if custom_prompt := self.config.get(f"{a_prefix}_prompt",
                                            "").strip(): sys_inst += f"\n\n【专属提示词】：\n{custom_prompt}"
        c_style = self.config.get("custom_style_prompt", "").strip()
        intro = intro_tpl.replace("{role}", b_role) + (
            f" 这是我的写作风格：\n{c_style}" if (c_style and "开发者" in role) else "")
        return is_tool, sys_inst, intro, mode_data.get("history", [])

    def pause(self):
        self.run_event.clear()

    def resume(self):
        self.run_event.set()

    def stop(self):
        self.is_running = False
        self.run_event.set()
        self.manual_review_event.set()

    def call_llm(self, prompt, role="Agent", use_fallback=False, history_text=""):
        self.run_event.wait()
        if not self.is_running: return None

        conf = self._get_api_config(use_fallback, role)
        self.log(f"[{role}] {conf['log_prefix']}思考中... ({conf['model_name']})")

        global_p = self.config.get("global_prompt", "").strip()
        b_prompt = f"\n{global_p}\n\n【任务】：\n{prompt}" if global_p else prompt
        is_tool, sys_inst, model_intro, pre_history = self._build_system_instructions(role)

        a_prefix = ROLE_MAP.get(role.split()[0].split('(')[0] if role else "", "")
        temperature, top_p, top_k = float(self.config.get(f"{a_prefix}_temperature", 0.7)), float(
            self.config.get(f"{a_prefix}_top_p", 0.9)), int(self.config.get(f"{a_prefix}_top_k", 40))

        safety_blocks = 0
        for attempt in range(MAX_API_RETRIES):
            try:
                self.run_event.wait()
                if not self.is_running: return None
                curr_prompt = f"{'虚拟艺术创作。' * safety_blocks}\n{b_prompt}" if safety_blocks else b_prompt

                api_key = self.key_manager.get_next_key(conf["api_keys_str"])
                if conf["api_type"] == "Gemini":
                    text = LLMService.call_gemini(api_key, conf["model_name"], sys_inst, curr_prompt, model_intro,
                                                  pre_history, history_text, temperature, top_p, top_k)
                else:
                    text = LLMService.call_openai(api_key, conf["api_url"], conf["model_name"], sys_inst, curr_prompt,
                                                  model_intro, pre_history, history_text, temperature, top_p)

                if not text: raise ValueError("API返回空文本")
                self.log(f"[{role}] 回复完成:\n{text[:50]}...\n" + "-" * 30)
                return text + ("\n❤" if use_fallback else "")
            except Exception as e:
                self.log(f"[{role}] ❌ 异常: {e}")
                self._save_error_log(role, conf, sys_inst, is_tool, model_intro, pre_history, curr_prompt, history_text,
                                     str(e))
                if "API未返回候选结果" in str(e) or "安全拦截" in str(e): safety_blocks += 1
                if attempt < MAX_API_RETRIES - 1: self.run_event.wait(2)

        if not use_fallback and self.config.get("fallback_api_keys", "").strip():
            self.log(f"[{role}] ⚠️ 临时切换备用API...")
            return self.call_llm(prompt, role, True, history_text)

        self.log(f"[{role}] ⛔ 彻底失败！暂停。")
        self.pause()
        self.run_event.wait()
        if self.is_running: return self.call_llm(prompt, role, False, history_text)
        return None

    def _extract_score(self, text):
        match = re.search(r'(?:分数|Score|得分)[:：]?\s*(\d{1,3})', text or "", re.IGNORECASE) or re.search(
            r'(?<!\d)(\d{1,3})(?!\d)', text or "")
        return int(match.group(1)) if match else None

    def _execute_parallel(self, limit, count, func, args_gen):
        with ThreadPoolExecutor(max_workers=max(1, min(limit, count))) as pool:
            return [f.result() for f in [pool.submit(func, *args_gen(i)) for i in range(count)]]

    def _evaluate_candidates(self, candidates, review_prompt_tpl, reviewer_role):
        if not candidates: return None, -1, 0

        # --- 人工评审环节逻辑 ---
        if not self.config.get("use_ai_reviewer", True):
            self.log(f"⏸ 触发人工评审环节 [{reviewer_role}] (共 {len(candidates)} 个候选方案等待确认)...")
            app_state.pending_candidates = candidates
            app_state.pending_review_type = reviewer_role
            app_state.manual_review_pending = True
            self.manual_review_event.clear()

            while not self.manual_review_event.is_set():
                if not self.is_running:
                    app_state.manual_review_pending = False
                    return None, 0, 0
                time.sleep(0.5)

            app_state.manual_review_pending = False
            self.log(f"✅ 人工评审完成，选用并修改了方案 {app_state.manual_review_selected_idx + 1}")
            return app_state.manual_review_result, 100, app_state.manual_review_selected_idx
        # ------------------------

        if len(candidates) == 1: return candidates[0], 100, 0
        rev_count = max(1, int(self.config.get("reviewer_count", 1)))

        en_role = ROLE_MAP.get(reviewer_role.split()[0].split('(')[0], "reviewer")
        hist_ctx = self._build_dynamic_history(int(self.config.get(f"{en_role}_full_ctx_count", 0))) if self.config.get(
            f"{en_role}_use_history", False) else ""

        scores_text = self._execute_parallel(len(candidates) * rev_count, len(candidates) * rev_count, self.call_llm,
                                             lambda i: (review_prompt_tpl.format(content=candidates[i // rev_count]),
                                                        f"{reviewer_role}(方案{i // rev_count + 1})", False, hist_ctx))

        scores = [sum(valid) / len(valid) if (valid := [s for t in scores_text[i * rev_count:(i + 1) * rev_count] if
                                                        (s := self._extract_score(t)) is not None]) else 0 for i in
                  range(len(candidates))]
        best_idx = scores.index(max(scores))
        self.log(f"🏆 选用 方案 {best_idx + 1}")
        return candidates[best_idx], scores[best_idx], best_idx

    def run_loop(self):
        self.is_running = True
        self.log(f"=== 启动生成 ({self.book_title}) ===")
        while self.is_running:
            if not self._step_design_phase() or not self._step_develop_phase() or not self._step_compress_phase(): continue
            if self._step_check_finish(): break
            self.current_chapter += 1

    def _step_design_phase(self):
        if self.use_manual:
            if not (info := self.manual_dict.get(self.current_chapter)):
                # 【关键修复点】：如果匹配不到大纲，抛出明确错误提示，不要静默死亡
                self.log(
                    f"❌ 找不到第 {self.current_chapter} 章的人工大纲，自动停止运行。请检查 UI 界面【人工制定各章大纲模式】中的章节号是否与当前的生成进度对应！")
                self.is_running = False
                return False
            self.best_plan = f"{info['name']}\n本章核心概要：{info['summary']}"
            return True

        self.log(f"\n--- 第 {self.current_chapter} 章 制定计划 ---")
        hist_ctx = self._build_dynamic_history(int(self.config.get("designer_full_ctx_count", 0))) if self.config.get(
            "designer_use_history", True) else ""

        plans = self._execute_parallel(int(self.config.get("designer_count", 1)),
                                       int(self.config.get("designer_count", 1)), self.call_llm, lambda i: (
                f"大纲：{self.config.get('outline', '')}。风格：{self.config.get('style', '')}。人物：{self.current_characters}。\n**仅为第{self.current_chapter}章**制定简短精炼计划。",
                f"设计者 {i + 1}", False, hist_ctx))
        if not plans or not self.is_running: return False
        self.best_plan, _, _ = self._evaluate_candidates(plans, "打分(0-100)，输出'分数: X'\n计划：{content}",
                                                         "评审者(计划)")
        return True

    def _step_develop_phase(self):
        self.log("\n--- 开发者编写正文 ---")
        hist_ctx = self._build_dynamic_history(int(self.config.get("developer_full_ctx_count", 20))) if self.config.get(
            "developer_use_history", True) else ""

        chapters = self._execute_parallel(int(self.config.get("developer_count", 1)),
                                          int(self.config.get("developer_count", 1)), self.call_llm, lambda i: (
                f"根据计划编写第{self.current_chapter}章正文(>2000字)，包裹在 <text> 和 </text> 内！\n：{self.best_plan}\n",
                f"开发者 {i + 1}", False, hist_ctx))
        if not chapters or not self.is_running: return False

        best_chapter, _, best_idx = self._evaluate_candidates(chapters,
                                                              f"打分(0-100)。\n计划：{self.best_plan}\n正文：{{content}}",
                                                              "评审者(打分)")

        if self.config.get("need_dev_revise", False):
            r_hist = self._build_dynamic_history(int(self.config.get("reviewer_full_ctx_count", 0))) if self.config.get(
                "reviewer_use_history", False) else ""
            feedback = self.call_llm(f"指出【正文】中瑕疵(勿夸奖)。\n计划：{self.best_plan}\n正文：{best_chapter}",
                                     "评审者(检视)", False, r_hist)

            j_hist = self._build_dynamic_history(int(self.config.get("judge_full_ctx_count", 0))) if self.config.get(
                "judge_use_history", False) else ""
            if self._extract_score(self.call_llm(f"打分(0-100)\n{feedback}", "裁判者", False, j_hist)) >= 80:
                best_chapter = self.call_llm(f"根据意见重修章节，包裹在 <text> 内！\n原：{best_chapter}\n意见：{feedback}",
                                             f"开发者 {best_idx + 1}", False, hist_ctx)

        if self.config.get("use_ai_cleaner", False):
            c_hist = self._build_dynamic_history(int(self.config.get("cleaner_full_ctx_count", 0))) if self.config.get(
                "cleaner_use_history", False) else ""
            clean_chap = self.call_llm(f"提取纯净小说正文：\n\n{best_chapter}", "清洗者(正文)", False, c_hist)
            final_text = clean_chap.strip() if clean_chap else best_chapter.strip()
        else:
            match = re.search(r'<text>(.*?)</text>', best_chapter, re.DOTALL | re.IGNORECASE)
            final_text = match.group(1).strip() if match else re.sub(r'^(好的|明白|以下是).*?\n', '',
                                                                     best_chapter).strip()

        final_text = re.sub(r'^\s*第[0-9一二三四五六七八九十百千万零〇]+[章节回].*?\n+', '', final_text).strip()

        if self.config.get("use_archiver", False):
            a_hist = self._build_dynamic_history(int(self.config.get("archiver_full_ctx_count", 0))) if self.config.get(
                "archiver_use_history", False) else ""
            archive_res = self.call_llm(
                f"分析正文是否有人物变化，无则输出【无需更新】，有则在原有格式增补。\n设定：{self.current_characters}\n正文：{final_text}",
                "归档者", False, a_hist)
            if archive_res and "无需更新" not in archive_res:
                self.current_characters = re.sub(r'^(好的|明白|没问题|以下是).*?\n', '', archive_res,
                                                 flags=re.IGNORECASE).strip().removeprefix("```json").removeprefix(
                    "```").removesuffix("```").strip()
                save_json(self.role_file, {"characters": self.current_characters})

        write_file(os.path.join(VolumeManager.get_chapter_dir(self.book_title), f"{self.current_chapter}.txt"),
                   final_text)
        self.chapter_texts[self.current_chapter] = final_text
        self.best_chapter = final_text
        self.full_volume = VolumeManager.rebuild_full(self.book_title)
        return True

    def compress_text(self, text, pass_history=True, protagonist_name=""):
        hist_ctx = self._build_dynamic_history(
            int(self.config.get("compressor_full_ctx_count", 0))) if pass_history and self.config.get(
            "compressor_use_history", False) else ""
        phint = f"主角是 {protagonist_name}，请用其真名。\n" if protagonist_name.strip() else ""

        summaries = self._execute_parallel(int(self.config.get("compressor_count", 1)),
                                           int(self.config.get("compressor_count", 1)), self.call_llm, lambda i: (
                f"设定：{self.current_characters}\n{phint}提取主线，包裹在 <summary> 和 </summary> 内！\n正文：\n{text}",
                f"压缩者 {i + 1}", False, hist_ctx))
        if not summaries or not self.is_running: return None

        best_sum, _, _ = self._evaluate_candidates(summaries, "打分(0-100)\n摘要：{content}", "评审者(压缩)")

        if self.config.get("use_ai_cleaner", False):
            c_hist = self._build_dynamic_history(int(self.config.get("cleaner_full_ctx_count", 0))) if self.config.get(
                "cleaner_use_history", False) else ""
            clean_sum = self.call_llm(f"提取纯净剧情摘要：\n\n{best_sum}", "清洗者(摘要)", False, c_hist)
            return clean_sum.strip() if clean_sum else best_sum.strip()

        match = re.search(r'<summary>(.*?)</summary>', best_sum, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else re.sub(r'^(好的|明白|以下是).*?\n', '', best_sum).strip()

    def _step_compress_phase(self):
        if not (final_sum := self.compress_text(self.best_chapter)): return False
        write_file(os.path.join(VolumeManager.get_summary_dir(self.book_title), f"{self.current_chapter}.txt"),
                   final_sum)
        self.summary_texts[self.current_chapter] = final_sum
        self.compressed_volume = VolumeManager.rebuild_compressed(self.book_title)
        self.save_local_data()
        return True

    def _step_check_finish(self):
        if self.target_chapter > 0 and self.current_chapter >= self.target_chapter:
            if self.use_manual and self.current_chapter >= self.max_manual_chapter:
                self.log("\n🎉 人工大纲完成！完结。")
            else:
                self.log(f"\n🎉 已生成至目标章节 (第 {self.target_chapter} 章)，自动停止。")
            self.is_running = False
            return True

        if self.use_manual:
            self.run_event.wait(2)
            return False

        hist_ctx = self._build_dynamic_history(int(self.config.get("designer_full_ctx_count", 0))) if self.config.get(
            "designer_use_history", True) else ""
        if "已完结" in (self.call_llm(
                f"当前摘要：{self.compressed_volume}。大纲：{self.config.get('outline', '')}。判断是否完结？只回答'已完结'或'未完结'。",
                "设计者(完结)", False, hist_ctx) or ""):
            self.log("\n🎉 小说已完结！");
            self.is_running = False;
            return True
        self.run_event.wait(2);
        return False


class AppState:
    def __init__(self):
        self.workflow, self.thread, self.logs, self.log_text = None, None, [], ""

        # 人工评审相关状态
        self.manual_review_pending = False
        self.pending_candidates = []
        self.pending_review_type = ""
        self.manual_review_result = ""
        self.manual_review_selected_idx = 0
        self.review_panel_shown = False

    def log_callback(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 500: self.logs = self.logs[-500:]
        self.log_text = "\n".join(self.logs)


app_state = AppState()