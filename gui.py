import re
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
from config import LocalModelConfig
from agent_controller import DebateOrchestrator
from model_utils import flush_cuda_cache

class Y2KCommunistGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Socialist AI Forum")
        self.root.geometry("850x650")

        # Win98 Style
        style = ttk.Style()
        style.theme_use('classic')

        self.bg_color = "#1e1e1e"
        self.frame_bg = "#2b2b2b"
        self.text_bg = "#121212"
        self.accent_red = "#b03030"
        self.accent_gold = "#d4af37"
        
        self.root.configure(bg=self.bg_color)
        self.config = LocalModelConfig()
        self.orchestrator = None
        self.msg_queue = queue.Queue()
        self.is_running = False
        self.batch_loading = False
        self.soft_limit_var = tk.BooleanVar(value=self.config.soft_token_limit)
        self.agent_extra_var = tk.StringVar(value=str(self.config.agent_extra_tokens))
        self.summary_extra_var = tk.StringVar(value=str(self.config.summary_extra_tokens))

        self._build_ui()
        self._build_hidden_window()
        self.root.after(100, self._process_queue)
        
        # Give UI time to draw before loading model
        self.root.after(500, self._load_model_bg)

    def _build_ui(self):
        # Top banner
        header_frame = tk.Frame(self.root, bg=self.frame_bg, bd=2, relief=tk.RAISED)
        header_frame.pack(fill=tk.X, padx=2, pady=2)
        header_label = tk.Label(header_frame, text="Socialist AI Forum", 
                                font=("MS Sans Serif", 14, "bold"), bg=self.frame_bg, fg=self.accent_gold)
        header_label.pack(pady=5)

        # Main content
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Sidebar for settings
        sidebar = tk.Frame(main_frame, bg=self.frame_bg, bd=2, relief=tk.RIDGE, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        tk.Label(sidebar, text="Parameters", font=("MS Sans Serif", 10, "bold"), bg=self.frame_bg, fg="#ffffff").pack(pady=(5,15))
        
        # Rounds
        tk.Label(sidebar, text="Debate Rounds:", bg=self.frame_bg, fg="#ffffff", font=("MS Sans Serif", 8)).pack(anchor=tk.W, padx=5)
        self.rounds_var = tk.StringVar(value=str(self.config.debate_rounds))
        tk.Spinbox(sidebar, from_=1, to=10, textvariable=self.rounds_var, font=("MS Sans Serif", 8), width=6).pack(anchor=tk.W, padx=5, pady=(0,15))

        # Max Tokens
        tk.Label(sidebar, text="Agent Response Length:", bg=self.frame_bg, fg="#ffffff", font=("MS Sans Serif", 8)).pack(anchor=tk.W, padx=5)
        self.tokens_var = tk.StringVar(value=str(self.config.agent_max_new_tokens))
        tk.Spinbox(sidebar, from_=50, to=1000, increment=50, textvariable=self.tokens_var, font=("MS Sans Serif", 8), width=6).pack(anchor=tk.W, padx=5, pady=(0,15))

        # Soft limit controls
        tk.Checkbutton(sidebar, text="Use soft limit", variable=self.soft_limit_var, bg=self.frame_bg, fg="#ffffff", selectcolor=self.frame_bg, activebackground=self.frame_bg, font=("MS Sans Serif", 8)).pack(anchor=tk.W, padx=5, pady=(0,8))
        tk.Label(sidebar, text="Soft limit buffer:", bg=self.frame_bg, fg="#ffffff", font=("MS Sans Serif", 8)).pack(anchor=tk.W, padx=5)
        self.agent_extra_spin = tk.Spinbox(sidebar, from_=0, to=200, increment=10, textvariable=self.agent_extra_var, font=("MS Sans Serif", 8), width=6)
        self.agent_extra_spin.pack(anchor=tk.W, padx=5, pady=(0,15))
        tk.Label(sidebar, text="Summary buffer:", bg=self.frame_bg, fg="#ffffff", font=("MS Sans Serif", 8)).pack(anchor=tk.W, padx=5)
        self.summary_extra_spin = tk.Spinbox(sidebar, from_=0, to=300, increment=10, textvariable=self.summary_extra_var, font=("MS Sans Serif", 8), width=6)
        self.summary_extra_spin.pack(anchor=tk.W, padx=5, pady=(0,15))

        # Batch toggle and status
        self.batch_button = tk.Button(sidebar, text="Batch Loading: OFF", font=("MS Sans Serif", 8, "bold"), 
                                     command=self._toggle_batch_mode, bg=self.frame_bg, fg="#ffffff", bd=2, activebackground="#3a3a3a")
        self.batch_button.pack(anchor=tk.W, padx=5, pady=(0,10), fill=tk.X)
        self.status_label = tk.Label(sidebar, text="Status: loading model...", fg="#ffffff", bg=self.frame_bg, font=("MS Sans Serif", 8, "bold"))
        self.status_label.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

        # Right Chat Area
        chat_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
        chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, bg=self.text_bg, fg="#e0e0e0", font=("Courier New", 10))
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        # Tag configurations for rich text terminal look
        self.chat_area.tag_config("system", foreground=self.accent_red, font=("Courier New", 10, "bold"))
        self.chat_area.tag_config("agent_name", foreground=self.accent_gold, font=("Courier New", 10, "bold"))
        self.chat_area.tag_config("agent_specialty", foreground="#a0a0a0", font=("Courier New", 10, "italic"))
        self.chat_area.tag_config("agent_text", foreground="#e0e0e0")
        self.chat_area.tag_config("summary_title", foreground=self.accent_gold, background=self.accent_red, font=("Courier New", 11, "bold"))
        self.chat_area.tag_config("summary_text", foreground="#e0e0e0", font=("Courier New", 10, "bold"))
        self.chat_area.tag_config("bold", font=("Courier New", 10, "bold"))
        self.chat_area.tag_config("italic", font=("Courier New", 10, "italic"))

        # Input Area Bottom
        input_frame = tk.Frame(self.root, bg=self.frame_bg, bd=2, relief=tk.RAISED)
        input_frame.pack(fill=tk.X, padx=2, pady=2)

        tk.Label(input_frame, text="Question:", font=("MS Sans Serif", 8, "bold"), bg=self.frame_bg, fg="#ffffff").pack(side=tk.LEFT, padx=5)
        
        self.input_entry = tk.Entry(input_frame, font=("MS Sans Serif", 10), bg="#2b2b2b", fg="#f0f0f0", insertbackground="#f0f0f0")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.input_entry.bind("<Return>", lambda e: self._on_submit())

        self.btn_submit = tk.Button(input_frame, text="Run Debate", font=("MS Sans Serif", 8, "bold"), 
                                    command=self._on_submit, bg=self.frame_bg, fg="#ffffff", bd=2, activebackground="#3a3a3a")
        self.btn_submit.pack(side=tk.RIGHT, padx=5, pady=5)

    def _build_hidden_window(self):
        self.hidden_window = tk.Toplevel(self.root)
        self.hidden_window.title("Hidden Reasoning")
        self.hidden_window.withdraw()
        self.hidden_text = scrolledtext.ScrolledText(self.hidden_window, wrap=tk.WORD, bg="#0f0f0f", fg="#888888", font=("Courier New", 10))
        self.hidden_text.pack(fill=tk.BOTH, expand=True)
        self.hidden_text.config(state=tk.DISABLED)

    def _append_plain(self, tag, text):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, text, tag)
        self.chat_area.config(state=tk.DISABLED)

    def _append_markdown(self, tag, text, scroll=True):
        text = text.replace("\r\n", "\n")
        text = re.sub(r'(?m)^\s*[-*]\s+', '• ', text)
        text = re.sub(r'(?m)^#{1,6}\s*(.*)$', r'\1\n', text)

        parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text)
        self.chat_area.config(state=tk.NORMAL)
        for part in parts:
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                self.chat_area.insert(tk.END, part[2:-2], (tag, "bold"))
            elif part.startswith("*") and part.endswith("*"):
                self.chat_area.insert(tk.END, part[1:-1], (tag, "italic"))
            else:
                self.chat_area.insert(tk.END, part, tag)
        if scroll:
            self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def _append_hidden_reasoning(self, text):
        self.hidden_text.config(state=tk.NORMAL)
        self.hidden_text.insert(tk.END, text)
        self.hidden_text.config(state=tk.DISABLED)

    def _append_chat(self, tag, text, scroll=True):
        if tag in {"agent_text", "summary_text"}:
            self._append_markdown(tag, text, scroll=scroll)
        else:
            self._append_plain(tag, text)

    def _set_batch_indicator(self, active: bool):
        if active:
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.delete("1.0", tk.END)
            self.chat_area.insert(tk.END, "[BATCH MODE] Generating responses, please wait...\n", "system")
            self.chat_area.config(state=tk.DISABLED)

    def _toggle_batch_mode(self):
        self.batch_loading = not self.batch_loading
        label = "Batch Loading: ON" if self.batch_loading else "Batch Loading: OFF"
        self.batch_button.config(text=label)

    def _load_model_bg(self):
        self._append_chat("system", "[SYSTEM] Loading model and preparing inference environment...\n")
        def load_task():
            try:
                self.orchestrator = DebateOrchestrator(self.config)
                self.msg_queue.put(("system", "[SYSTEM] Model is ready.\n\n"))
                self.msg_queue.put(("ready", None))
            except Exception as e:
                self.msg_queue.put(("system", f"[ERROR] {str(e)}\n"))
        threading.Thread(target=load_task, daemon=True).start()

    def _on_submit(self):
        if self.is_running or not self.orchestrator:
            return
        
        question = self.input_entry.get().strip()
        if not question:
            return
            
        self.input_entry.delete(0, tk.END)
        self.is_running = True
        self.btn_submit.config(state=tk.DISABLED)
        self.status_label.config(text="Status: processing debate...", fg="#000080")
        if self.batch_loading:
            self._set_batch_indicator(True)
        
        # Sync simple settings
        try:
            self.config.debate_rounds = int(self.rounds_var.get())
            self.config.agent_max_new_tokens = int(self.tokens_var.get())
            self.config.soft_token_limit = bool(self.soft_limit_var.get())
            self.config.agent_extra_tokens = int(self.agent_extra_var.get())
            self.config.summary_extra_tokens = int(self.summary_extra_var.get())
        except:
            pass
            
        def debate_task():
            try:
                if self.batch_loading:
                    self.orchestrator.run_debate_batch(question, lambda evt, payload: self.msg_queue.put((evt, payload)))
                else:
                    self.orchestrator.run_debate_stream(question, lambda evt, payload: self.msg_queue.put((evt, payload)))
            except Exception as e:
                self.msg_queue.put(("system", f"\n[ERROR] {str(e)}\n"))
                self.msg_queue.put(("done", None))
            finally:
                flush_cuda_cache()

        threading.Thread(target=debate_task, daemon=True).start()

    def _process_queue(self):
        try:
            # Process multiple items per tick to prevent lagging behind fast token streams
            for _ in range(50):
                event_type, payload = self.msg_queue.get_nowait()
                if event_type == "ready":
                    self.status_label.config(text="Status: ready", fg="#008000")
                elif event_type == "system":
                    self._append_chat("system", payload)
                elif event_type == "agent_name":
                    self._append_chat("agent_name", payload, scroll=False)
                elif event_type == "agent_specialty":
                    self._append_chat("agent_specialty", payload)
                elif event_type == "agent_text":
                    self._append_chat("agent_text", payload)
                    self.root.update_idletasks()
                elif event_type == "summary_title":
                    self._append_chat("summary_title", payload)
                elif event_type == "summary_text":
                    self._append_chat("summary_text", payload)
                elif event_type == "hidden_reasoning":
                    self._append_hidden_reasoning(payload)
                elif event_type == "done":
                    self.is_running = False
                    self.btn_submit.config(state=tk.NORMAL)
                    self.status_label.config(text="Status: ready", fg="#008000")
                    if self.batch_loading:
                        self._set_batch_indicator(False)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = Y2KCommunistGUI(root)
    root.mainloop()
