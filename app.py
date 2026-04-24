# author: YilEnS e1351599@u.nus.edu
# app.py - 前端 UI 交互层

import gradio as gr
import threading, time, os
import backend


# ==========================================
# 前端包装与状态映射方法 (UI Wrappers)
# ==========================================

def update_choices(items):
    return gr.update(choices=items, value=items[0] if items else None)


def get_books_ui():
    return update_choices(backend.FileManager.get_books())


def get_manual_outlines_ui():
    return update_choices(backend.FileManager._get_list("outline"))


def get_roles_ui():
    return update_choices(backend.FileManager._get_list("roles"))


def save_mode_ui(mode_name, sys_prompt, intro, history_data):
    msg, modes = backend.ModeManager.save_mode(mode_name, sys_prompt, intro, history_data)
    updates = [gr.update(choices=modes) for _ in range(len(backend.AGENT_NAMES_MAP))]
    return [msg] + updates


def export_book_folder_ui(title):
    path, msg = backend.FileManager.export_book_folder(title)
    return gr.update(value=path, visible=bool(path)), msg


def delete_book_ui(title):
    msg, new_books = backend.FileManager.delete_book(title)
    return msg, update_choices(new_books)


def load_book_config_ui(raw_title):
    if not raw_title: return gr.update(), gr.update(), gr.update(), "⚠️ 请输入书名"
    conf = backend.load_json(backend.get_dir("Books", backend.safe_name(raw_title.strip()), "book_config.json"))
    if conf:
        return gr.update(value=conf.get("outline", "")), gr.update(value=conf.get("style", "")), gr.update(
            value=conf.get("characters", "")), f"✅ 加载《{raw_title}》配置成功"
    return gr.update(), gr.update(), gr.update(), "⚠️ 未找到专属配置"


def fetch_models_ui(api_type, url_str, keys_str):
    models, msg = backend.fetch_models(api_type, url_str, keys_str)
    if models: return gr.update(choices=models, value=models[0]), msg
    return gr.update(), msg


def on_app_load_ui():
    try:
        backend.mode.init_modes(backend.get_dir("modes"))
    except AttributeError:
        pass  # Handle if mode module does not have init_modes

    modes = backend.get_all_modes()
    conf = backend.load_json(backend.get_dir("user_input.json"))

    res = []
    for k, d_val in backend.ALL_CONFIG_KEYS:
        val = conf.get(k, d_val)
        if k.endswith("_mode"):
            val = val if val in modes else (d_val if d_val in modes else (modes[0] if modes else None))
            res.append(gr.update(choices=modes, value=val))
        else:
            res.append(gr.update(value=val))
    return tuple(res)


def start_generation_ui(*args):
    config = backend.build_config_dict(*args)
    if not config["book_title"].strip() or not config["api_keys"].strip():
        yield backend.app_state.log_text + "\n❌ 错误: 书名和主API Key不能为空！", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return
    if not config["use_manual_outline"] and not config["outline"].strip():
        yield backend.app_state.log_text + "\n❌ 错误: 未启用人工大纲时，总大纲不能为空！", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    backend.save_config_json(config)
    if backend.app_state.workflow and not backend.app_state.workflow.run_event.is_set():
        backend.app_state.workflow.resume()
        backend.app_state.log_callback("▶ 恢复生成...")
    else:
        backend.app_state.logs = []
        backend.app_state.workflow = backend.AgentWorkflow(config, backend.app_state.log_callback)
        backend.app_state.thread = threading.Thread(target=backend.app_state.workflow.run_loop, daemon=True)
        backend.app_state.thread.start()

    while backend.app_state.workflow and (backend.app_state.workflow.is_running or backend.app_state.thread.is_alive()):
        is_paused = not backend.app_state.workflow.run_event.is_set()
        btn_text = "▶ 继续" if is_paused else "⏸ 暂停"

        # 处理人工评审阶段 UI 的展示逻辑
        if backend.app_state.manual_review_pending:
            if not getattr(backend.app_state, 'review_panel_shown', False):
                choices = [f"方案 {i + 1}" for i in range(len(backend.app_state.pending_candidates))]
                backend.app_state.review_panel_shown = True
                yield (backend.app_state.log_text,
                       gr.update(value=btn_text, interactive=True),
                       gr.update(visible=True),
                       f"**正在评审环节**: {backend.app_state.pending_review_type}",
                       gr.update(choices=choices, value=choices[0]),
                       backend.app_state.pending_candidates[0])
            else:
                yield backend.app_state.log_text, gr.update(value=btn_text,
                                                            interactive=True), gr.update(), gr.update(), gr.update(), gr.update()
        else:
            backend.app_state.review_panel_shown = False
            yield backend.app_state.log_text, gr.update(value=btn_text, interactive=True), gr.update(
                visible=False), gr.update(), gr.update(), gr.update()

        time.sleep(0.5)

    yield backend.app_state.log_text, gr.update(value="▶ 开始生成", interactive=True), gr.update(
        visible=False), gr.update(), gr.update(), gr.update()


def toggle_pause_ui():
    if not backend.app_state.workflow: return "▶ 开始生成", "尚未启动工作流。"
    if backend.app_state.workflow.run_event.is_set():
        backend.app_state.workflow.pause()
        backend.app_state.log_callback("⏸ 暂停挂起...")
        return "▶ 继续", "已发送暂停指令"
    backend.app_state.workflow.resume()
    backend.app_state.log_callback("▶ 恢复生成...")
    return "⏸ 暂停", "工作流已恢复"


def import_manual_outline_ui(f):
    if not f: return gr.update(), "❌ 未选择文件"
    data, msg = backend.ImportManager.import_manual_outline(f.name)
    return data, msg


def import_novel_ui(file_obj, title, protagonist, *config_args):
    if not file_obj or not title.strip():
        yield "❌ 缺少上传的文件或目标书名"
        return
    config = backend.build_config_dict(*config_args)
    for msg in backend.ImportManager.import_novel(file_obj.name, title, protagonist, config):
        yield msg


def save_chapter_ui(raw_title, chapter_str, new_content, *config_args):
    config = backend.build_config_dict(*config_args)
    final_sum, msg = backend.EditManager.save_chapter(raw_title, chapter_str, new_content, config)
    if final_sum:
        return final_sum, msg
    return gr.update(), msg


# 响应选择更改：前端动态刷新人工评审文本
def on_review_choice_change(choice):
    if not choice or not backend.app_state.pending_candidates: return ""
    try:
        idx = int(choice.split(" ")[1]) - 1
        return backend.app_state.pending_candidates[idx]
    except:
        return backend.app_state.pending_candidates[0]


# 提交人工评审修改
def submit_manual_review(choice, text):
    if not choice:
        idx = 0
    else:
        try:
            idx = int(choice.split(" ")[1]) - 1
        except:
            idx = 0
    backend.app_state.manual_review_result = text
    backend.app_state.manual_review_selected_idx = idx
    if backend.app_state.workflow:
        backend.app_state.workflow.manual_review_event.set()
    return gr.update(visible=False), "✅ 人工评审已提交，工作流继续运行..."


# ==========================================
# 前端 UI 构建逻辑
# ==========================================
def build_ui():
    with gr.Blocks(title="VibeNoving v0.4.7 nightly test verion", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 📚 VibeNoving v0.4.7 nightly test verion")

        with gr.Tabs():
            with gr.TabItem("📥 导入半成品小说"):
                gr.Markdown("### 📥 仅支持章节名为 第x章的小说")
                with gr.Row():
                    import_file = gr.File(label="上传 .txt 小说文件", file_types=[".txt"])
                    import_title = gr.Textbox(label="设定新书名（上传后自动提取或手动输入）")
                import_protagonist = gr.Textbox(label="主角姓名（可选，用于辅助AI在提取第一人称摘要时准确识别主角真名）")
                import_btn = gr.Button("🚀 开始智能导入与自动处理", variant="primary")
                import_status = gr.Textbox(label="导入进度状态", lines=5)

            with gr.TabItem("✍️ 剧情与设定输入"):
                with gr.Row():
                    book_title = gr.Textbox(label="书名 (必选)", scale=4)
                    btn_load_book = gr.Button("📂 加载该书历史设定", scale=1)
                load_book_msg = gr.Markdown("")
                with gr.Row():
                    outline = gr.Textbox(label="剧情总大纲", lines=5)
                    btn_outline = gr.UploadButton("上传", file_types=[".txt"])
                with gr.Row():
                    style = gr.Textbox(label="剧情风格 (可选)", lines=3)
                    btn_style = gr.UploadButton("上传", file_types=[".txt"])
                with gr.Row():
                    characters = gr.Textbox(label="人物列表 (可选)", lines=3)
                    btn_char = gr.UploadButton("上传", file_types=[".txt"])

                for btn, comp in [(btn_outline, outline), (btn_style, style), (btn_char, characters)]:
                    btn.upload(lambda f: backend.read_file(f.name) if f else "", inputs=[btn], outputs=[comp])

                with gr.Accordion("🎭 人物设定库 (保存与加载)", open=False):
                    with gr.Row():
                        role_dropdown = gr.Dropdown(label="设定文件")
                        btn_load_role, btn_refresh_roles = gr.Button("📂 加载"), gr.Button("🔄 刷新")
                    with gr.Row():
                        role_save_name, btn_save_role = gr.Textbox(label="设定名"), gr.Button("💾 保存为新设定",
                                                                                              variant="primary")
                    role_op_msg = gr.Markdown("")

                with gr.Accordion("📝 人工制定各章大纲模式 (可选)", open=False):
                    use_manual_outline = gr.Checkbox(label="启用人工输入大纲")
                    with gr.Row():
                        outline_dropdown = gr.Dropdown(label="本地大纲")
                        btn_load_outline, btn_refresh_outlines = gr.Button("📂 加载"), gr.Button("🔄 刷新")
                    with gr.Row():
                        outline_save_name = gr.Textbox(label="大纲名称")
                        btn_save_outline = gr.Button("💾 保存到本地", variant="primary")
                        btn_import_outline_txt = gr.UploadButton("📄 导入TXT大纲并切片", file_types=[".txt"])

                    outline_op_msg = gr.Markdown("")
                    manual_outline_data = gr.Dataframe(headers=["章节号", "章节名", "章节概要"],
                                                       datatype=["number", "str", "str"], col_count=(3, "fixed"),
                                                       row_count=(1, "dynamic"), interactive=True, type="array")
                    btn_add_manual_row = gr.Button("➕ 新增一章大纲", size="sm")

            with gr.TabItem("🤖 Agent 综合专属配置"):
                agent_inputs_dict = {}
                with gr.Accordion("➕ 新建全局模式预设", open=False):
                    new_mode_name, new_mode_sys, new_mode_intro = gr.Textbox(label="预设名"), gr.Textbox(
                        label="Agent人设"), gr.Textbox(label="自我介绍")
                    new_mode_history = gr.Dataframe(headers=["用户输入", "模型回复"], datatype=["str", "str"],
                                                    col_count=(2, "fixed"), row_count=(1, "dynamic"), value=[["", ""]],
                                                    interactive=True)
                    btn_add_history_row, btn_save_mode, mode_save_msg = gr.Button("➕ 新增", size="sm"), gr.Button(
                        "💾 保存", variant="primary"), gr.Markdown("")

                api_status_agent = gr.Textbox(label="获取状态", interactive=False)
                with gr.Tabs():
                    for zh, en, _ in backend.AGENT_NAMES_MAP:
                        with gr.TabItem(f"{zh} ({en})"):
                            agent_prompt = gr.Textbox(label=f"【0】{zh} 专属额外提示词", lines=2)
                            with gr.Row():
                                agent_mode, agent_count = gr.Dropdown(label=f"【1】加载配置", choices=[]), gr.Number(
                                    label=f"【2】并发数", precision=0)

                            gr.Markdown("### 📚 【3】读取前文策略")
                            with gr.Row():
                                agent_use_history = gr.Checkbox(label="读取前文",
                                                                value=(en in ["designer", "developer"]))
                                agent_full_ctx_count = gr.Number(label="完整章数",
                                                                 value=(20 if en == "developer" else 0), precision=0)

                            with gr.Row():
                                agent_temp, agent_topp, agent_topk = gr.Slider(0.0, 2.0, step=0.1,
                                                                               label="Temp"), gr.Slider(0.0, 1.0,
                                                                                                        step=0.05,
                                                                                                        label="Top P"), gr.Slider(
                                    1, 100, step=1, label="Top K")

                            gr.Markdown("### ⚙️ 【4】独立 API (留空默认全局)")
                            with gr.Row():
                                a_type = gr.Dropdown(["Gemini", "OpenAI Compatible"], label="API 类型")
                                a_keys = gr.Textbox(label="API Keys(逗号隔开)", type="password",
                                                    placeholder="输入 Key...")
                            with gr.Row():
                                a_url = gr.Textbox(label="API URL (非Gemini必填)",
                                                   placeholder="如https://integrate.api.nvidia.com/v1")
                                a_model = gr.Dropdown(label="模型选择", allow_custom_value=True, interactive=True)
                                btn_f = gr.Button("🔄 获取模型列表")
                            btn_f.click(fetch_models_ui, inputs=[a_type, a_url, a_keys],
                                        outputs=[a_model, api_status_agent])
                            agent_inputs_dict[en] = {"mode": agent_mode, "prompt": agent_prompt, "count": agent_count,
                                                     "use_history": agent_use_history,
                                                     "full_ctx_count": agent_full_ctx_count,
                                                     "temp": agent_temp, "top_p": agent_topp, "top_k": agent_topk,
                                                     "api_type": a_type, "api_keys": a_keys, "api_url": a_url,
                                                     "api_model": a_model}

            with gr.TabItem("⚙️ 基础&全局API配置"):
                with gr.Row():
                    with gr.Column(scale=2):
                        global_prompt = gr.Textbox(label="全局提示词(可传入世界观)")
                        custom_style_prompt = gr.Textbox(label="自定义风格")
                    with gr.Column(scale=1):
                        use_ai_reviewer = gr.Checkbox(label="开启AI评审(如果不开的话，每章都由用户评审并修改)", value=True)
                        need_dev_revise = gr.Checkbox(label="开发者修改文本(暂不推荐开启)")
                        use_ai_cleaner = gr.Checkbox(label="AI提取正文(暂不推荐开启)")
                        use_archiver = gr.Checkbox(label="暂不推荐开启")
                        context_max_chars = gr.Number(label="前文最大截取字数", precision=0)

                api_status_main = gr.Textbox(label="全局API获取状态", interactive=False)
                gr.Markdown("### 【主 API 配置】")
                with gr.Row():
                    api_type = gr.Dropdown(["Gemini", "OpenAI Compatible"], label="主 API 类型")
                    api_keys = gr.Textbox(label="主 API Keys", type="password")
                with gr.Row():
                    api_url, api_model, btn_f_main = gr.Textbox(label="主 API URL"), gr.Dropdown(label="主模型选择",
                                                                                                 allow_custom_value=True), gr.Button(
                        "🔄 获取主模型")
                btn_f_main.click(fetch_models_ui, inputs=[api_type, api_url, api_keys],
                                 outputs=[api_model, api_status_main])

                gr.Markdown("### 【备用 API 配置】")
                with gr.Row():
                    fallback_api_type, fallback_api_keys = gr.Dropdown(["Gemini", "OpenAI Compatible"],
                                                                       label="备用 API 类型"), gr.Textbox(
                        label="备用 API Keys", type="password")
                with gr.Row():
                    fallback_api_url, fallback_api_model, btn_f_fall = gr.Textbox(label="备用 API URL"), gr.Dropdown(
                        label="备用模型选择", allow_custom_value=True), gr.Button("🔄 获取备用模型")
                btn_f_fall.click(fetch_models_ui, inputs=[fallback_api_type, fallback_api_url, fallback_api_keys],
                                 outputs=[fallback_api_model, api_status_main])

            with gr.TabItem("💻 控制台 & 日志"):
                with gr.Row():
                    btn_start = gr.Button("▶ 开始", variant="primary")
                    btn_pause = gr.Button("⏸ 暂停", interactive=False)
                    btn_save_conf = gr.Button("💾 保存配置")
                    target_chapter = gr.Number(label="生成到第几章 (留空0默认生成所有章节)", precision=0)
                sys_msg = gr.Textbox(label="系统提示")
                log_output = gr.Textbox(label="运行日志", lines=25, max_lines=25)

                # 新增人工评审专属面板，当系统挂起时显示
                with gr.Group(visible=False) as review_panel:
                    gr.Markdown("### 🙋‍♂️ 人工评审环节 (系统已挂起，等待您选择与修改)")
                    review_type_msg = gr.Markdown("")
                    with gr.Row():
                        review_choices = gr.Radio(label="查看生成的候选方案", choices=[])
                        btn_review_submit = gr.Button("✅ 确认使用修改后的内容并继续", variant="primary")
                    review_text = gr.Textbox(label="在此编辑您最喜欢的内容，它将被直接采用", lines=15)

            with gr.TabItem("📁 文件与章节管理"):
                btn_refresh, book_select = gr.Button("🔄 刷新"), gr.Dropdown(label="小说")
                with gr.Row():
                    btn_download, btn_delete = gr.Button("📦 下载", variant="primary"), gr.Button("🗑️ 删除",
                                                                                                 variant="stop")
                fm_msg, fm_file = gr.Textbox(label="状态"), gr.File(label="下载", visible=False)

                gr.Markdown("--- \n ### 📝 本地章节内容覆写")
                with gr.Row():
                    edit_chapter_select, btn_load_ch = gr.Dropdown(label="章节"), gr.Button("📂 获取")
                edit_status = gr.Markdown("")
                with gr.Row():
                    edit_ch_cont, edit_ch_sum = gr.Textbox(label="正文", lines=10), gr.Textbox(label="摘要", lines=10)
                btn_save_ch = gr.Button("💾 保存修改并触发AI重新压缩", variant="primary")

        # ---------------------------
        # 收集变量
        # ---------------------------
        base_inputs = [book_title, outline, style, characters, use_manual_outline, manual_outline_data, global_prompt,
                       custom_style_prompt, use_ai_reviewer, need_dev_revise, use_ai_cleaner, use_archiver,
                       context_max_chars,
                       target_chapter,
                       api_type, api_keys, api_url, api_model, fallback_api_type, fallback_api_keys, fallback_api_url,
                       fallback_api_model]
        agent_inputs_list = []
        for _, en, _ in backend.AGENT_NAMES_MAP:
            d = agent_inputs_dict[en]
            agent_inputs_list.extend(
                [d["mode"], d["prompt"], d["count"], d["use_history"], d["full_ctx_count"], d["temp"], d["top_p"],
                 d["top_k"], d["api_type"], d["api_keys"], d["api_url"], d["api_model"]])
        all_inputs = base_inputs + agent_inputs_list

        # ---------------------------
        # 绑定事件
        # ---------------------------
        demo.load(on_app_load_ui, inputs=[], outputs=all_inputs)
        demo.load(get_books_ui, inputs=[], outputs=[book_select])
        demo.load(get_manual_outlines_ui, inputs=[], outputs=[outline_dropdown])
        demo.load(get_roles_ui, inputs=[], outputs=[role_dropdown])

        btn_add_manual_row.click(lambda df: df + [[len(df) + 1, "", ""]] if df else [[1, "", ""]],
                                 inputs=[manual_outline_data], outputs=[manual_outline_data])
        btn_add_history_row.click(lambda df: df + [["", ""]] if df else [["", ""]], inputs=[new_mode_history],
                                  outputs=[new_mode_history])

        btn_load_book.click(load_book_config_ui, inputs=[book_title],
                            outputs=[outline, style, characters, load_book_msg])
        btn_save_mode.click(save_mode_ui, inputs=[new_mode_name, new_mode_sys, new_mode_intro, new_mode_history],
                            outputs=[mode_save_msg] + [agent_inputs_dict[en]["mode"] for _, en, _ in
                                                       backend.AGENT_NAMES_MAP])
        btn_save_conf.click(lambda *args: backend.save_config_json(backend.build_config_dict(*args)), inputs=all_inputs,
                            outputs=[sys_msg])

        btn_save_outline.click(lambda n, d: (msg := backend.FileManager._save_json_ui("outline", n, d, "大纲")[0],
                                             get_manual_outlines_ui()), inputs=[outline_save_name, manual_outline_data],
                               outputs=[outline_op_msg, outline_dropdown])
        btn_import_outline_txt.upload(
            import_manual_outline_ui,
            inputs=[btn_import_outline_txt],
            outputs=[manual_outline_data, outline_op_msg]
        )
        btn_load_outline.click(
            lambda f: (res := backend.FileManager._load_json_ui("outline", f), res[0] or gr.update(), res[1]),
            inputs=[outline_dropdown], outputs=[manual_outline_data, outline_op_msg])
        btn_refresh_outlines.click(get_manual_outlines_ui, inputs=[], outputs=[outline_dropdown])

        btn_save_role.click(
            lambda n, d: (msg := backend.FileManager._save_json_ui("roles", n, {"characters": d}, "人物设定")[0],
                          get_roles_ui()), inputs=[role_save_name, characters], outputs=[role_op_msg, role_dropdown])
        btn_load_role.click(
            lambda f: (res := backend.FileManager._load_json_ui("roles", f, "characters"), res[0] or gr.update(),
                       res[1]), inputs=[role_dropdown], outputs=[characters, role_op_msg])
        btn_refresh_roles.click(get_roles_ui, inputs=[], outputs=[role_dropdown])

        btn_start.click(start_generation_ui, inputs=all_inputs,
                        outputs=[log_output, btn_pause, review_panel, review_type_msg, review_choices, review_text])
        btn_pause.click(toggle_pause_ui, inputs=[], outputs=[btn_pause, sys_msg])

        # 人工评审相关事件绑定
        review_choices.change(on_review_choice_change, inputs=[review_choices], outputs=[review_text])
        btn_review_submit.click(submit_manual_review, inputs=[review_choices, review_text],
                                outputs=[review_panel, sys_msg])

        btn_refresh.click(get_books_ui, inputs=[], outputs=[book_select])
        btn_download.click(export_book_folder_ui, inputs=[book_select], outputs=[fm_file, fm_msg])
        btn_delete.click(delete_book_ui, inputs=[book_select], outputs=[fm_msg, book_select])

        import_file.upload(lambda f: os.path.splitext(os.path.basename(f.name))[0] if f else "", inputs=[import_file],
                           outputs=[import_title])
        import_btn.click(import_novel_ui, inputs=[import_file, import_title, import_protagonist] + all_inputs,
                         outputs=[import_status])

        book_select.change(lambda r: update_choices(backend.EditManager.get_generated_chapters(r)),
                           inputs=[book_select], outputs=[edit_chapter_select])
        btn_load_ch.click(backend.EditManager.load_chapter, inputs=[book_select, edit_chapter_select],
                          outputs=[edit_ch_cont, edit_ch_sum, edit_status])
        btn_save_ch.click(save_chapter_ui, inputs=[book_select, edit_chapter_select, edit_ch_cont] + all_inputs,
                          outputs=[edit_ch_sum, edit_status])

        demo.load(None, inputs=None, outputs=None,
                  js='''() => { alert("温馨提示：这是一个内测版本，功能尚在完善中。有问题请发送邮件至e1351599@u.nus.edu"); }''')

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)