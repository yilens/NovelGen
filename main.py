import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import json
import os
import time
import re
from google import genai
from google.genai import types

# --- 常量与文件路径 ---
CONFIG_FILE = "user_input.json"
FULL_VOLUME_FILE = "full_volume.txt"
COMPRESSED_VOLUME_FILE = "compressed_volume.txt"


class AgentWorkflow:
    def __init__(self, config, log_callback):
        self.config = config
        self.log = log_callback
        self.api_keys = [k.strip() for k in config.get("api_keys", "").split(",") if k.strip()]
        self.key_index = 0
        self.is_paused = False
        self.is_running = False

        # 运行时状态
        self.full_volume = ""
        self.compressed_volume = ""
        self.current_chapter = 1
        self.load_local_data()

    def load_local_data(self):
        """加载本地保存的完整卷和压缩卷"""
        if os.path.exists(FULL_VOLUME_FILE):
            with open(FULL_VOLUME_FILE, "r", encoding="utf-8") as f:
                self.full_volume = f.read()
        if os.path.exists(COMPRESSED_VOLUME_FILE):
            with open(COMPRESSED_VOLUME_FILE, "r", encoding="utf-8") as f:
                self.compressed_volume = f.read()

    def save_local_data(self):
        """保存完整卷和压缩卷到本地"""
        with open(FULL_VOLUME_FILE, "w", encoding="utf-8") as f:
            f.write(self.full_volume)
        with open(COMPRESSED_VOLUME_FILE, "w", encoding="utf-8") as f:
            f.write(self.compressed_volume)

    def get_next_client(self):
        """API轮询：获取下一个GenAI Client"""
        if not self.api_keys:
            raise ValueError("未配置API Keys")
        key = self.api_keys[self.key_index]
        self.key_index = (self.key_index + 1) % len(self.api_keys)

        # 如果需要自定义Base URL，可在http_options中配置 (视google-genai版本而定)
        client = genai.Client(api_key=key)
        return client

    def call_llm(self, prompt, role="Agent", retry_count=0):
        """调用LLM，包含降级审核、重试、自动暂停和5秒休眠逻辑"""
        while getattr(self, 'is_paused', False):
            time.sleep(1)
            if getattr(self, 'is_running', False) is False: return None

        model_name = self.config.get("api_model", "gemini-2.5-flash")
        self.log(f"[{role}] 正在思考... (Model: {model_name})")

        try:
            client = self.get_next_client()

            # --- 新增：自动拼接全局提示词 ---
            global_prompt = self.config.get("global_prompt", "").strip()
            if global_prompt:
                # 使用特殊的括号强化全局指令的优先级
                final_prompt = f"【系统最高优先级指令（覆盖一切默认规则）】：\n{global_prompt}\n\n【你的当前具体任务】：\n{prompt}"
            else:
                final_prompt = prompt

            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]

            response = client.models.generate_content(
                model=model_name,
                contents=final_prompt,  # <--- 注意这里改成了 final_prompt
                config=types.GenerateContentConfig(safety_settings=safety_settings)
            )

            if not response.candidates:
                raise ValueError("API未返回任何候选结果(可能被平台安全拦截或网络异常)")

            candidate = response.candidates[0]

            if candidate.finish_reason and "STOP" not in str(candidate.finish_reason):
                raise ValueError(f"内容生成异常终止，原因: {candidate.finish_reason} (触发了安全审核或其他截断)")

            text = response.text
            if not text:
                raise ValueError("API返回了空文本")

            time.sleep(5)
            return text

        except Exception as e:
            self.log(f"[{role}] ❌ API调用异常: {e}")
            time.sleep(5)

            if retry_count < 10:
                self.log(f"[{role}] 🔄 准备尝试下一个 API Key 重新作业...")
                return self.call_llm(prompt, role, retry_count=retry_count + 1)
            else:
                self.log(f"[{role}] ⛔ 连续两次调用失败！工作流已自动暂停。")
                self.is_paused = True

                # 【修复点 2】：安全调用 pause_callback，防止部分更新导致的 AttributeError
                pause_cb = getattr(self, 'pause_callback', None)
                if pause_cb:
                    pause_cb(True)

                while getattr(self, 'is_paused', False):
                    time.sleep(1)
                    if getattr(self, 'is_running', False) is False: return None

                self.log(f"[{role}] ▶ 工作流已恢复，重新尝试调用...")
                return self.call_llm(prompt, role, retry_count=0)

    def extract_score(self, text):
        """从评审者或裁判者的回复中提取0-100的分数"""
        if not text:  # 防御性拦截：如果传入 None，直接给默认分
            return 50

        match = re.search(r'(?:分数|Score|得分)[:：]?\s*(\d{1,3})', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 50

    def run_loop(self):
        """执行14步核心工作流"""
        self.is_running = True
        self.log("=== 启动 AI 小说自动生成闭环 ===")

        # 简化版大纲数据
        outline = self.config.get("outline", "")
        style = self.config.get("style", "")
        chars = self.config.get("characters", "")

        while self.is_running:
            # 步骤 1: 设计者制定计划
            self.log("\n--- 步骤 1: 设计者制定总体开发计划 ---")
            designer_count = int(self.config.get("designer_count", 1))
            plans = []
            for i in range(designer_count):
                prompt = f"你是设计者。大纲：{outline}。风格：{style}。人物：{chars}。请为当前第{self.current_chapter}章及后续章节制定剧情发展计划，每章需>2000字且结尾留有悬念。"
                plan = self.call_llm(prompt, f"设计者 {i + 1}")
                plans.append(plan)

            # 步骤 2: 评审者对计划评分
            self.log("\n--- 步骤 2: 评审者选出最佳开发计划 ---")
            reviewer_count = int(self.config.get("reviewer_count", 1))
            best_plan = plans[0]
            if len(plans) > 1:
                # 简化逻辑：直接让第一个评审者打分选出最高
                highest_score = -1
                for plan in plans:
                    score_prompt = f"你是评审者。请对以下剧情计划打分(0-100)。只输出：'分数: X'。\n计划：{plan}"
                    res = self.call_llm(score_prompt, "评审者(计划)")
                    score = self.extract_score(res)
                    if score > highest_score:
                        highest_score = score
                        best_plan = plan
            self.log("已选定最佳开发计划。")

            # 步骤 3 & 4: 开发者根据计划开发、自测试并修改
            self.log("\n--- 步骤 3 & 4: 开发者编写章节并自验证 ---")
            dev_count = int(self.config.get("developer_count", 1))
            chapters = []
            for i in range(dev_count):
                prompt = f"你是开发者。请根据以下计划编写第{self.current_chapter}章的正文内容（不少于2000字）。\n历史摘要：{self.compressed_volume}\n开发计划：{best_plan}\n写完后请自行验证并输出修改后的最终版。"
                chapter_text = self.call_llm(prompt, f"开发者 {i + 1}")
                chapters.append(chapter_text)

            # 步骤 5 & 6: 评审者对章节评分并提出检视意见
            self.log("\n--- 步骤 5 & 6: 评审者评分并提出检视意见 ---")
            best_chapter = chapters[0]
            best_dev_index = 0
            highest_score = -1
            for i, chapter in enumerate(chapters):
                score_prompt = f"你是评审者。请对以下章节进行严格客观的打分(0-100)。输出'分数: X'。\n章节内容：{chapter}"
                res = self.call_llm(score_prompt, f"评审者(打分 开发者{i + 1})")
                score = self.extract_score(res)
                if score > highest_score:
                    highest_score = score
                    best_chapter = chapter
                    best_dev_index = i

            feedback_prompt = f"你是评审者。请对以下选出的最高分章节提出你不喜欢部分的检视修改意见。\n章节内容：{best_chapter}"
            feedback = self.call_llm(feedback_prompt, "评审者(检视意见)")
            self.log(f"评审者对最佳章节提出了检视意见。")

            # 步骤 7: 开发者查看意见并决定是否上诉
            self.log("\n--- 步骤 7: 开发者查看意见 ---")
            rebuttal_prompt = f"你是开发者。评审者对你的章节提出了以下意见：\n{feedback}\n你是否认同？如果不认同请说明理由。如果认同请回答'完全认同'。"
            dev_reply = self.call_llm(rebuttal_prompt, f"开发者 {best_dev_index + 1}")

            needs_revision = True
            if "完全认同" not in dev_reply:
                # 步骤 8: 裁判者判定
                self.log("\n--- 步骤 8: 开发者提出异议，裁判者进行判定 ---")
                judge_prompt = f"你是裁判者。开发者和评审者发生分歧。评审意见：{feedback}\n开发者反驳：{dev_reply}\n请对检视意见的合理性打分(0-100)。输出'分数: X'。"
                res = self.call_llm(judge_prompt, "裁判者")
                judge_score = self.extract_score(res)
                self.log(f"裁判者打分: {judge_score}")
                if judge_score < 70:
                    needs_revision = False
                    self.log("裁判者判定：无需修改。")
                else:
                    self.log("裁判者判定：检视意见合理，必须修改。")

            # 步骤 9 & 10: 开发者修改并合入完整卷
            if needs_revision:
                self.log("\n--- 步骤 9: 开发者进行最终修改 ---")
                revise_prompt = f"你是开发者。请根据以下必须修改的检视意见，重新修改你的章节。\n原章节：{best_chapter}\n意见：{feedback}"
                best_chapter = self.call_llm(revise_prompt, f"开发者 {best_dev_index + 1}")

            self.log("\n--- 步骤 10: 清洗者提取纯净正文并合入完整卷 ---")
            cleaner_prompt_chapter = f"你是清洗者。你的唯一任务是提取文本中的纯净小说正文。请去除以下文本中所有的AI寒暄、开头语（如'好的'、'作为开发者'）、以及不属于小说正文的标题和解释性文字。绝对不要输出'这是清洗后的文本'等废话，只输出干净的小说正文：\n\n{best_chapter}"
            clean_chapter = self.call_llm(cleaner_prompt_chapter, "清洗者(正文)")

            # 双重保险：LLM清洗后，再用原有的正则轻度去一下首尾空白
            clean_chapter = clean_chapter.strip() if clean_chapter else best_chapter.strip()
            self.full_volume += f"\n\n第{self.current_chapter}章\n{clean_chapter}"

            # 步骤 11 & 12: 压缩者总结新章节，评审者选优合入压缩卷
            self.log("\n--- 步骤 11 & 12: 压缩者生成摘要并评审 ---")
            comp_count = int(self.config.get("compressor_count", 1))
            summaries = []
            for i in range(comp_count):
                comp_prompt = f"你是压缩者。请对以下最新一章内容进行剧情压缩概括：\n{best_chapter}"
                summary = self.call_llm(comp_prompt, f"压缩者 {i + 1}")
                summaries.append(summary)

            best_summary = summaries[0]
            if len(summaries) > 1:
                highest_score = -1
                for summary in summaries:
                    score_prompt = f"你是评审者。请对剧情摘要打分(0-100)。输出'分数: X'。\n摘要：{summary}"
                    res = self.call_llm(score_prompt, "评审者(压缩评分)")
                    score = self.extract_score(res)
                    if score > highest_score:
                        highest_score = score
                        best_summary = summary

            # --- 在合入压缩卷之前，增加清洗摘要的逻辑 ---
            self.log("\n--- 清洗者提取纯净摘要并合入压缩卷 ---")
            cleaner_prompt_summary = f"你是清洗者。你的唯一任务是提取文本中的纯净剧情摘要。请去除以下文本中所有的AI寒暄、开头语（如'好的，设计者'）、以及多余的标题（如'【剧情压缩概括】'）。绝对不要输出任何多余的对话，只输出纯粹的剧情概括文本：\n\n{best_summary}"
            clean_summary = self.call_llm(cleaner_prompt_summary, "清洗者(摘要)")

            clean_summary = clean_summary.strip() if clean_summary else best_summary.strip()
            self.compressed_volume += f"\n第{self.current_chapter}卷剧情：{clean_summary}"
            self.save_local_data()
            self.log(f"第 {self.current_chapter} 章内容已持久化保存。")

            # 步骤 13: 清空工作区
            self.log("\n--- 步骤 13: 新章节开发完成，清空Agent工作区 ---")
            # 在程序逻辑中，局部变量(plans, chapters等)在下一次循环自动释放

            # 步骤 14: 设计者判断是否完结
            self.log("\n--- 步骤 14: 设计者判断是否完结 ---")
            finish_prompt = f"你是设计者。当前小说摘要：{self.compressed_volume}。根据大纲：{outline}，请判断小说是否已经完结？只回答'已完结'或'未完结'。"
            finish_res = self.call_llm(finish_prompt, "设计者(完结判断)")

            if "已完结" in finish_res:
                self.log("\n🎉 小说已完结！生成停止。")
                self.is_running = False
                break
            else:
                self.current_chapter += 1
                self.log(f"未完结，准备开始第 {self.current_chapter} 章的开发...")
                time.sleep(2)

    def stop(self):
        self.is_running = False


class NovelGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自闭环AI小说生成器")
        self.root.geometry("800x600")

        self.workflow = None
        self.thread = None
        self.create_widgets()
        self.load_config()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # 1. 用户输入 Tab
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="用户输入 (剧情/设定)")
        self.build_input_tab(input_frame)

        # 2. Agent 设置 Tab
        agent_frame = ttk.Frame(notebook)
        notebook.add(agent_frame, text="开发组设置 (数量/提示词)")
        self.build_agent_tab(agent_frame)

        # 3. API 设置 Tab
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API 配置")
        self.build_api_tab(api_frame)

        # 4. 控制与日志 Tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="控制台 & 日志")
        self.build_log_tab(log_frame)

    def build_input_tab(self, frame):
        ttk.Label(frame, text="剧情大纲 (必选):").grid(row=0, column=0, sticky='w')
        self.outline_text = scrolledtext.ScrolledText(frame, height=5)
        self.outline_text.grid(row=1, column=0, columnspan=2, sticky='ew')
        ttk.Button(frame, text="上传 .txt", command=lambda: self.load_txt(self.outline_text)).grid(row=0, column=1,
                                                                                                   sticky='e')

        ttk.Label(frame, text="剧情风格 (可选):").grid(row=2, column=0, sticky='w')
        self.style_text = scrolledtext.ScrolledText(frame, height=3)
        self.style_text.grid(row=3, column=0, columnspan=2, sticky='ew')
        ttk.Button(frame, text="上传 .txt", command=lambda: self.load_txt(self.style_text)).grid(row=2, column=1,
                                                                                                 sticky='e')

        ttk.Label(frame, text="人物列表 (可选):").grid(row=4, column=0, sticky='w')
        self.char_text = scrolledtext.ScrolledText(frame, height=3)
        self.char_text.grid(row=5, column=0, columnspan=2, sticky='ew')
        ttk.Button(frame, text="上传 .txt", command=lambda: self.load_txt(self.char_text)).grid(row=4, column=1,
                                                                                                sticky='e')

    def build_agent_tab(self, frame):
        # --- 新增：全局提示词输入区 ---
        ttk.Label(frame, text="全局系统提示词\n(所有Agent生效):").grid(row=0, column=0, sticky='nw', pady=5)
        self.global_prompt_text = scrolledtext.ScrolledText(frame, height=4, width=50)
        self.global_prompt_text.grid(row=0, column=1, sticky='ew', pady=5)

        # --- 原有的Agent设置，行号整体下移 ---
        roles = ["设计者 (Designer)", "开发者 (Developer)", "评审者 (Reviewer)", "裁判者 (Judge)", "压缩者 (Compressor)", "清洗者 (Cleaner)"]
        self.agent_vars = {}
        for i, role in enumerate(roles):
            base_row = (i * 2) + 1  # 从第1行开始往下排
            ttk.Label(frame, text=f"{role} 数量:").grid(row=base_row, column=0, sticky='w', pady=(5,0))
            count_var = tk.StringVar(value="1")
            ttk.Entry(frame, textvariable=count_var, width=5).grid(row=base_row, column=1, sticky='w', pady=(5,0))
            ttk.Label(frame, text="预设提示词:").grid(row=base_row+1, column=0, sticky='w')
            prompt_var = tk.StringVar()
            ttk.Entry(frame, textvariable=prompt_var, width=50).grid(row=base_row+1, column=1, sticky='w')
            role_key = role.split()[0]
            self.agent_vars[role_key] = {"count": count_var, "prompt": prompt_var}

    def build_api_tab(self, frame):
        ttk.Label(frame, text="API Keys (必选, 逗号分隔用于轮询):").grid(row=0, column=0, sticky='w', pady=5)
        self.api_keys_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.api_keys_var, width=60).grid(row=1, column=0, columnspan=2, sticky='w')

        ttk.Label(frame, text="API 请求网址 (必选):").grid(row=2, column=0, sticky='w', pady=5)
        self.api_url_var = tk.StringVar(value="https://generativelanguage.googleapis.com")
        ttk.Entry(frame, textvariable=self.api_url_var, width=60).grid(row=3, column=0, columnspan=2, sticky='w')

        ttk.Label(frame, text="API 模型 (必选):").grid(row=4, column=0, sticky='w', pady=5)
        self.api_model_var = tk.StringVar(value="gemini-2.5-flash")
        ttk.Entry(frame, textvariable=self.api_model_var, width=30).grid(row=5, column=0, sticky='w')

    def build_log_tab(self, frame):
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)

        self.start_btn = ttk.Button(control_frame, text="▶ 开始生成", command=self.start_generation)
        self.start_btn.pack(side='left', padx=5)

        self.pause_btn = ttk.Button(control_frame, text="⏸ 暂停", command=self.pause_generation, state=tk.DISABLED)
        self.pause_btn.pack(side='left', padx=5)

        self.save_btn = ttk.Button(control_frame, text="💾 保存配置", command=self.save_config)
        self.save_btn.pack(side='right', padx=5)

        self.log_text = scrolledtext.ScrolledText(frame, state='disabled')
        self.log_text.pack(fill='both', expand=True)

    def load_txt(self, text_widget):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, content)

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def get_config_dict(self):
        # 增加 cleaner_count 的保存
        return {
            "global_prompt": self.global_prompt_text.get(1.0, tk.END).strip(),  # 新增这行
            "outline": self.outline_text.get(1.0, tk.END).strip(),
            "style": self.style_text.get(1.0, tk.END).strip(),
            "characters": self.char_text.get(1.0, tk.END).strip(),
            "api_keys": self.api_keys_var.get(),
            "api_url": self.api_url_var.get(),
            "api_model": self.api_model_var.get(),
            "designer_count": self.agent_vars["设计者"]["count"].get(),
            "developer_count": self.agent_vars["开发者"]["count"].get(),
            "reviewer_count": self.agent_vars["评审者"]["count"].get(),
            "judge_count": self.agent_vars["裁判者"]["count"].get(),
            "compressor_count": self.agent_vars["压缩者"]["count"].get(),
            "cleaner_count": self.agent_vars["清洗者"]["count"].get(),  # 新增
        }

    def save_config(self):
        config = self.get_config_dict()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("成功", "配置已保存到本地")

    def load_config(self):
        # 增加 cleaner_count 的加载
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                if hasattr(self, 'global_prompt_text'):
                    self.global_prompt_text.delete(1.0, tk.END)
                    self.global_prompt_text.insert(tk.END, config.get("global_prompt", ""))
                self.outline_text.insert(tk.END, config.get("outline", ""))
                self.style_text.insert(tk.END, config.get("style", ""))
                self.char_text.insert(tk.END, config.get("characters", ""))
                self.api_keys_var.set(config.get("api_keys", ""))
                self.api_url_var.set(config.get("api_url", "https://generativelanguage.googleapis.com"))
                self.api_model_var.set(config.get("api_model", "gemini-2.5-flash"))

                for role, vals in self.agent_vars.items():
                    if role == "设计者": vals["count"].set(config.get("designer_count", "1"))
                    if role == "开发者": vals["count"].set(config.get("developer_count", "1"))
                    if role == "评审者": vals["count"].set(config.get("reviewer_count", "1"))
                    if role == "裁判者": vals["count"].set(config.get("judge_count", "1"))
                    if role == "压缩者": vals["count"].set(config.get("compressor_count", "1"))
                    if role == "清洗者": vals["count"].set(config.get("cleaner_count", "1"))  # 新增

    def start_generation(self):
        if not self.api_keys_var.get().strip():
            messagebox.showerror("错误", "API Keys 不能为空！")
            return
        if not self.outline_text.get(1.0, tk.END).strip():
            messagebox.showerror("错误", "剧情大纲不能为空！")
            return

        self.save_config()
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="⏸ 暂停")

        if self.workflow and self.workflow.is_paused:
            self.workflow.is_paused = False
            self.log_message("▶ 恢复生成...")
        else:
            self.workflow = AgentWorkflow(self.get_config_dict(), self.log_message)
            self.thread = threading.Thread(target=self.workflow.run_loop, daemon=True)
            self.thread.start()

    def pause_generation(self):
        if self.workflow:
            if not self.workflow.is_paused:
                self.workflow.is_paused = True
                self.pause_btn.config(text="▶ 继续")
                self.log_message("⏸ 已暂停。等待当前API请求完成后挂起。")
            else:
                self.workflow.is_paused = False
                self.pause_btn.config(text="⏸ 暂停")
                self.log_message("▶ 恢复生成...")


if __name__ == "__main__":
    root = tk.Tk()
    app = NovelGeneratorGUI(root)
    root.mainloop()