# author: YilEnS e1351599@u.nus.edu
# backend.py - 后端核心逻辑层

import json, os, re, time, shutil, threading, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import mode  # Agent配置文件
import tools  # 工具层
import api  # API网络层

# ==========================================
# 常量配置与全局初始化
# ==========================================
MAX_API_RETRIES = 10

ROLE_MAP = {
    "设计者": "designer", "开发者": "developer", "评审者": "reviewer",
    "裁判者": "judge", "压缩者": "compressor", "归档者": "archiver"
}

AGENT_NAMES_MAP = [
    ("设计者", "designer", "Designer_R16.json"),
    ("开发者", "developer", "Developer_R16.json"),
    ("评审者", "reviewer", "Reviewer_R16.json"),
    ("裁判者", "judge", "Judger_R16.json"),
    ("压缩者", "compressor", "Compressor_R16.json"),
    ("归档者", "archiver", "Archiver_R16.json")
]

BASE_CONFIG_KEYS = [
    ("book_title", ""), ("outline", ""), ("style", ""), ("characters", ""),
    ("use_manual_outline", False), ("manual_outline_data", [[1, "", ""]]),
    ("global_prompt", ""), ("custom_style_prompt", ""),
    ("designer_use_manual_review", False), ("developer_use_manual_review", False),
    ("compressor_use_manual_review", False),
    ("need_dev_revise", False), ("use_archiver", False),
    ("context_max_chars", 50000), ("target_chapter", 0), ("vibenoving_context_count", 5),
    ("api_type", "Gemini"), ("api_keys", ""), ("api_url", ""), ("api_model", ""),
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
        (f"{en}_api_model", "")
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
    def open_books_folder():
        folder_path = os.path.abspath(get_dir("Books"))
        os.makedirs(folder_path, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
            return f"✅ 已在本地文件管理器中打开文件夹: {folder_path}"
        except Exception as e:
            return f"❌ 无法打开文件夹: {e}"

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
            cn2arb = {'零': 0, '〇': 0, '一': 1, '两': 2, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                      '九': 9}
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

            actual_ch_num = ImportManager.parse_chapter_number(ch_title_raw, ch_idx_fallback)

            outline_data.append([actual_ch_num, ch_title_clean, ch_content])
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
            # 强制取消保存时的压缩端人工审查，防止后端死锁挂起
            config["compressor_use_manual_review"] = False
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

    @staticmethod
    def run_vibenoving(raw_title, chapter_str, current_text, selected_text, user_prompt, config):
        """局部重写核心逻辑(VibeNoving)"""
        if not selected_text.strip(): return "❌ 请先选中并填入需要修改的原文片段。"
        if not user_prompt.strip(): return "❌ 请输入修改方向。"

        ch_num = 1
        if chapter_str and isinstance(chapter_str, str):
            match = re.search(r'\d+', chapter_str)
            if match: ch_num = int(match.group())
        elif hasattr(app_state, 'workflow') and app_state.workflow:
            ch_num = app_state.workflow.current_chapter

        vibe_ctx_count = int(config.get("vibenoving_context_count", 5))
        history = ""
        if vibe_ctx_count > 0 and raw_title:
            sum_dir = VolumeManager.get_summary_dir(raw_title)
            sums = []
            for i in range(max(1, ch_num - vibe_ctx_count), ch_num):
                sf = os.path.join(sum_dir, f"{i}.txt")
                if os.path.exists(sf):
                    sums.append(f"第{i}章摘要：{read_file(sf)}")
            if sums:
                history = "【前文背景】\n" + "\n".join(sums)

        sys_prompt = "你是一个专业的小说编辑和润色助手。你需要根据前文背景和用户的具体要求，对小说片段进行重写或润色。\n\n**严格遵守以下规则**：\n1. 你【只能】输出修改后的这部分片段文本，不要包含原章节的其它未修改段落。\n2. 绝不能输出任何解释、寒暄或总结性的话语。\n3. 必须保持原有的人称和核心设定。"
        llm_prompt = f"{history}\n\n【本章当前完整内容参考】\n{current_text}\n\n【需要你修改的原文片段】\n{selected_text}\n\n【修改方向/要求】\n{user_prompt}\n\n请直接输出修改后的片段文本："

        api_type = config.get("api_type", "Gemini")
        api_keys = config.get("api_keys", "")
        api_url = config.get("api_url", "")
        api_model = config.get("api_model", "")

        if not api_keys:
            return "❌ 未配置全局 API Key，请在设置中配置。"

        try:
            km = api.APIKeyManager()
            api_key = km.get_next_key(api_keys)
            if api_type == "Gemini":
                res = api.LLMService.call_gemini(api_key, api_model, sys_prompt, llm_prompt,
                                                 "收到，我将仅返回修改后的文本段落。", [], "", 0.7, 0.9, 40)
            else:
                res = api.LLMService.call_openai(api_key, api_url, api_model, sys_prompt, llm_prompt,
                                                 "收到，我将仅返回修改后的文本段落。", [], "", 0.7, 0.9)
            return str(res).strip()
        except Exception as e:
            return f"❌ VibeNoving 生成失败: {e}"


# ==========================================
# 核心业务对象与工作流
# ==========================================
class AgentWorkflow:
    def __init__(self, config, log_callback):
        self.config, self.log = config, log_callback
        self.key_manager = api.APIKeyManager()
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

        self.update_config(self.config)

    def update_config(self, new_config):
        """支持运行时动态更新设置，并重新计算目标章节"""
        self.config = new_config
        self.log("🔄 已实时更新系统配置。")
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
                if self.target_chapter <= 0:
                    self.target_chapter = self.max_manual_chapter
                else:
                    self.target_chapter = min(max(self.current_chapter, self.target_chapter), self.max_manual_chapter)
                self.log(f"🎯 设定当前生成目标至：第 {self.target_chapter} 章 (人工大纲模式)")
        else:
            if self.target_chapter > 0:
                self.log(f"🎯 设定当前生成目标至：第 {self.target_chapter} 章")

    def _build_dynamic_history(self, full_count):
        actual_full = max(0, min(full_count, self.current_chapter - 1))
        start_full = self.current_chapter - actual_full

        history_parts, summary_parts, full_parts = [], [], []

        # 修复老缓存问题：拼接前文时优先强行读取最新的本地文件，防止工作流内存不同步
        sum_dir = VolumeManager.get_summary_dir(self.book_title)
        for i in range(1, start_full):
            sf = os.path.join(sum_dir, f"{i}.txt")
            if os.path.exists(sf):
                summary_parts.append(f"第{i}章剧情：{read_file(sf)}")
            elif i in self.summary_texts:
                summary_parts.append(f"第{i}章剧情：{self.summary_texts[i]}")

        if summary_parts: history_parts.append("【前文剧情摘要】\n" + "\n".join(summary_parts))

        ch_dir = VolumeManager.get_chapter_dir(self.book_title)
        for i in range(start_full, self.current_chapter):
            cf = os.path.join(ch_dir, f"{i}.txt")
            if os.path.exists(cf):
                full_parts.append(f"第{i}章\n{read_file(cf)}")
            elif i in self.chapter_texts:
                full_parts.append(f"第{i}章\n{self.chapter_texts[i]}")

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
            return {"model_name": self.config.get("fallback_api_model", ""),
                    "api_type": self.config.get("fallback_api_type", "Gemini"),
                    "api_url": self.config.get("fallback_api_url", "").strip(),
                    "api_keys_str": self.config.get("fallback_api_keys", ""), "log_prefix": "【备用API】"}

        b_role = role.split()[0].split('(')[0] if role else ""
        a_prefix = ROLE_MAP.get(b_role, "")
        if a_prefix and self.config.get(f"{a_prefix}_api_keys", "").strip():
            return {"model_name": self.config.get(f"{a_prefix}_api_model", ""),
                    "api_type": self.config.get(f"{a_prefix}_api_type", "Gemini"),
                    "api_url": self.config.get(f"{a_prefix}_api_url", "").strip(),
                    "api_keys_str": self.config.get(f"{a_prefix}_api_keys", ""), "log_prefix": f"【{b_role}专属】"}

        return {"model_name": self.config.get("api_model", ""),
                "api_type": self.config.get("api_type", "Gemini"),
                "api_url": self.config.get("api_url", "").strip(), "api_keys_str": self.config.get("api_keys", ""),
                "log_prefix": ""}

    def _build_system_instructions(self, role):
        b_role = role.split()[0].split('(')[0]
        is_tool = "压缩者" in role or "归档者" in role
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

        # --- 新增/修复逻辑：将全局提示词和人物设定进行高权重全局注入 ---
        global_p = self.config.get("global_prompt", "").strip()
        ctx_parts = []
        if global_p:
            ctx_parts.append(f"【全局世界观与提示】\n{global_p}")
        if hasattr(self, 'current_characters') and self.current_characters.strip():
            ctx_parts.append(f"【当前人物与背景设定】\n{self.current_characters.strip()}")

        ctx_text = "\n\n".join(ctx_parts)
        b_prompt = f"{ctx_text}\n\n【当前任务】\n{prompt}" if ctx_text else f"【当前任务】\n{prompt}"
        # --------------------------------------------------------

        is_tool, sys_inst, model_intro, pre_history = self._build_system_instructions(role)

        b_role = role.split()[0].split('(')[0] if role else ""
        en_role = ROLE_MAP.get(b_role, "")

        a_prefix = ROLE_MAP.get(b_role, "")
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
                    result = api.LLMService.call_gemini(api_key, conf["model_name"], sys_inst, curr_prompt, model_intro,
                                                        pre_history, history_text, temperature, top_p, top_k,
                                                        role_key=en_role)
                else:
                    result = api.LLMService.call_openai(api_key, conf["api_url"], conf["model_name"], sys_inst,
                                                        curr_prompt,
                                                        model_intro, pre_history, history_text, temperature, top_p,
                                                        role_key=en_role)

                # 判断为空的宽容处理（应对Tool返回的字典或数字结果）
                if result is None or (isinstance(result, str) and not result): raise ValueError("API返回空内容")
                self.log(f"[{role}] 回复完成:\n{str(result)[:50]}...\n" + "-" * 30)

                # 如果是字符串且触发了备用模型逻辑，才拼接心形图标
                if use_fallback and isinstance(result, str):
                    result += "\n❤"
                return result
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
        # 兼容处理 Tools 模型直出的 Integer
        if isinstance(text, (int, float)):
            return int(text)
        if isinstance(text, dict) and "score" in text:
            return int(text["score"])
        text_str = str(text)
        match = re.search(r'(?:分数|Score|得分)[:：]?\s*(\d{1,3})', text_str, re.IGNORECASE) or \
                re.search(r'(?<!\d)(\d{1,3})(?!\d)', text_str)
        return int(match.group(1)) if match else None

    def _execute_parallel(self, limit, count, func, args_gen):
        with ThreadPoolExecutor(max_workers=max(1, min(limit, count))) as pool:
            return [f.result() for f in [pool.submit(func, *args_gen(i)) for i in range(count)]]

    def _evaluate_candidates(self, candidates, review_prompt_tpl, reviewer_role):
        if not candidates: return None, -1, 0

        # --- 人工评审环节逻辑 ---
        needs_manual = False
        if "计划" in reviewer_role:
            needs_manual = self.config.get("designer_use_manual_review", False)
        elif "打分" in reviewer_role:
            needs_manual = self.config.get("developer_use_manual_review", False)
        elif "压缩" in reviewer_role:
            needs_manual = self.config.get("compressor_use_manual_review", False)

        if needs_manual:
            self.log(f"⏸ 触发人工评审环节 [{reviewer_role}] (共 {len(candidates)} 个候选方案等待确认)...")
            app_state.pending_candidates = candidates
            app_state.pending_review_type = reviewer_role
            app_state.manual_review_pending = True
            app_state.manual_review_retry = False
            self.manual_review_event.clear()

            while not self.manual_review_event.is_set():
                if not self.is_running:
                    app_state.manual_review_pending = False
                    return None, 0, 0
                time.sleep(0.5)

            app_state.manual_review_pending = False

            # 捕获用户是否点击了“重试”
            if getattr(app_state, 'manual_review_retry', False):
                self.log(f"🔄 用户选择了重试当前生成环节 [{reviewer_role}]...")
                return "_RETRY_", -1, -1

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
                self.log(
                    f"❌ 找不到第 {self.current_chapter} 章的人工大纲，自动停止运行。请检查 UI 界面【人工制定各章大纲模式】中的章节号是否与当前的生成进度对应！")
                self.is_running = False
                return False
            self.best_plan = f"{info['name']}\n本章核心概要：{info['summary']}"
            return True

        self.log(f"\n--- 第 {self.current_chapter} 章 制定计划 ---")
        hist_ctx = self._build_dynamic_history(int(self.config.get("designer_full_ctx_count", 0))) if self.config.get(
            "designer_use_history", True) else ""

        # 利用 while 循环支持用户反复重试本环节
        while True:
            # 优化了大纲和风格的排版，人物设定已交由 call_llm 全局注入
            outline_text = self.config.get('outline', '').strip()
            style_text = self.config.get('style', '').strip()

            task_prompt = ""
            if outline_text:
                task_prompt += f"【小说总大纲】\n{outline_text}\n\n"
            if style_text:
                task_prompt += f"【期望风格】\n{style_text}\n\n"

            task_prompt += f"请结合以上信息与前文，**仅为第{self.current_chapter}章**制定简短精炼的剧情计划。"

            plans = self._execute_parallel(int(self.config.get("designer_count", 1)),
                                           int(self.config.get("designer_count", 1)), self.call_llm, lambda i: (
                    task_prompt,
                    f"设计者 {i + 1}", False, hist_ctx))
            if not plans or not self.is_running: return False
            self.best_plan, _, _ = self._evaluate_candidates(plans, "打分(0-100)，输出'分数: X'\n计划：{content}",
                                                             "评审者(计划)")
            if self.best_plan == "_RETRY_": continue
            break
        return True

    def _step_develop_phase(self):
        self.log("\n--- 开发者编写正文 ---")
        hist_ctx = self._build_dynamic_history(int(self.config.get("developer_full_ctx_count", 20))) if self.config.get(
            "developer_use_history", True) else ""

        while True:
            # 修复核心问题：给开发者注入前置设定要求，配合 call_llm 中的高权重人物列表注入
            chapters = self._execute_parallel(int(self.config.get("developer_count", 1)),
                                              int(self.config.get("developer_count", 1)), self.call_llm, lambda i: (
                    f"请根据以下计划编写第{self.current_chapter}章正文(建议2000字以上)。请务必遵循人物设定与前文逻辑。\n\n【本章计划】：\n{self.best_plan}\n",
                    f"开发者 {i + 1}", False, hist_ctx))
            if not chapters or not self.is_running: return False

            best_chapter, _, best_idx = self._evaluate_candidates(chapters,
                                                                  f"打分(0-100)。\n计划：{self.best_plan}\n正文：{{content}}",
                                                                  "评审者(打分)")
            if best_chapter == "_RETRY_": continue
            break

        if self.config.get("need_dev_revise", False):
            r_hist = self._build_dynamic_history(int(self.config.get("reviewer_full_ctx_count", 0))) if self.config.get(
                "reviewer_use_history", False) else ""
            feedback = self.call_llm(f"指出【正文】中瑕疵(勿夸奖)。\n计划：{self.best_plan}\n正文：{best_chapter}",
                                     "评审者(检视)", False, r_hist)

            j_hist = self._build_dynamic_history(int(self.config.get("judge_full_ctx_count", 0))) if self.config.get(
                "judge_use_history", False) else ""
            if self._extract_score(self.call_llm(f"打分(0-100)\n{feedback}", "裁判者", False, j_hist)) >= 80:
                best_chapter = self.call_llm(f"根据意见重修章节。\n原：{best_chapter}\n意见：{feedback}",
                                             f"开发者 {best_idx + 1}", False, hist_ctx)

        final_text = str(best_chapter).strip()

        if self.config.get("use_archiver", False):
            a_hist = self._build_dynamic_history(int(self.config.get("archiver_full_ctx_count", 0))) if self.config.get(
                "archiver_use_history", False) else ""
            archive_res = self.call_llm(
                f"对比全局【当前人物与背景设定】，分析以下正文是否导致了人物状态变化或新人物登场。无则输出【无需更新】，有则在原有格式上增补并输出最新设定。\n\n【本章正文】：\n{final_text}",
                "归档者", False, a_hist)

            if isinstance(archive_res, dict):
                if archive_res.get("status") in ["需要更新", "更新", "有变化"] and archive_res.get(
                        "updated_characters"):
                    self.current_characters = archive_res["updated_characters"]
                    save_json(self.role_file, {"characters": self.current_characters})
            elif isinstance(archive_res, str) and "无需更新" not in archive_res:
                clean_text = archive_res.replace("```json", "").replace("```", "").strip()
                if clean_text:
                    self.current_characters = clean_text
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

        while True:
            summaries = self._execute_parallel(int(self.config.get("compressor_count", 1)),
                                               int(self.config.get("compressor_count", 1)), self.call_llm, lambda i: (
                    f"{phint}请提取以下正文的剧情主线摘要。\n\n【正文】：\n{text}",
                    f"压缩者 {i + 1}", False, hist_ctx))
            if not summaries or not self.is_running: return None

            best_sum, _, _ = self._evaluate_candidates(summaries, "打分(0-100)\n摘要：{content}", "评审者(压缩)")
            if best_sum == "_RETRY_": continue

            return str(best_sum).strip()

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
        if "已完结" in (str(self.call_llm(
                f"当前摘要：{self.compressed_volume}。大纲：{self.config.get('outline', '')}。判断是否完结？只回答'已完结'或'未完结'。",
                "设计者(完结)", False, hist_ctx)) or ""):
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

        # 重试标识位
        self.manual_review_retry = False

    def log_callback(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 500: self.logs = self.logs[-500:]
        self.log_text = "\n".join(self.logs)


app_state = AppState()