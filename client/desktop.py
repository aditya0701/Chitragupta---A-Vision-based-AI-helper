"""Chitragupt Desktop App — a standalone Tkinter-based GUI client."""

from __future__ import annotations
import asyncio
import base64
import io
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from tkinter.font import Font

import httpx
from PIL import Image, ImageTk

API_URL = "http://localhost:8000"


class ChitraguptDesktop:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chitragupt — Vision Assistant")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

        # Dark theme colors
        self.bg = "#0d0d1a"
        self.fg = "#e0e0e0"
        self.accent = "#3a7bd5"
        self.accent2 = "#00d2ff"
        self.card_bg = "#1a1a2e"
        self.border = "#2a2a4a"

        self.root.configure(bg=self.bg)
        self.current_image_base64 = None
        self.current_image_path = None
        self.api_client = httpx.AsyncClient(timeout=120.0)

        self._setup_styles()
        self._build_ui()
        self._check_health()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.bg)
        style.configure("TLabel", background=self.bg, foreground=self.fg)
        style.configure("TButton", background=self.card_bg, foreground=self.fg, borderwidth=1, focusthickness=0)
        style.map("TButton", background=[("active", self.accent)])

    def _build_ui(self):
        # ─── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#1a1a2e", height=52)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        title = tk.Label(
            header, text="✦  Chitragupt", font=("Segoe UI", 18, "bold"),
            bg="#1a1a2e", fg=self.accent2,
        )
        title.pack(side="left", padx=16, pady=10)

        self.status_label = tk.Label(
            header, text="● Checking...", font=("Segoe UI", 10),
            bg="#1a1a2e", fg="#888",
        )
        self.status_label.pack(side="right", padx=16, pady=10)

        # ─── Main Content ────────────────────────────────────────────────────
        main_frame = tk.Frame(self.root, bg=self.bg)
        main_frame.pack(fill="both", expand=True)

        # Chat display
        chat_frame = tk.Frame(main_frame, bg=self.bg)
        chat_frame.pack(fill="both", expand=True, padx=12, pady=(12, 0))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap="word",
            font=("Segoe UI", 12),
            bg=self.card_bg,
            fg=self.fg,
            insertbackground=self.fg,
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=12,
            state="disabled",
        )
        self.chat_display.pack(fill="both", expand=True)

        # Configure text tags
        self.chat_display.tag_config("user", foreground="#4da6ff", font=("Segoe UI", 12, "bold"))
        self.chat_display.tag_config("assistant", foreground=self.fg)
        self.chat_display.tag_config("system", foreground="#ffaa00", font=("Segoe UI", 10))
        self.chat_display.tag_config("tool", foreground="#ffaa00", font=("Segoe UI", 10))
        self.chat_display.tag_config("model", foreground="#666", font=("Segoe UI", 9))

        # ─── Image Preview ───────────────────────────────────────────────────
        self.image_frame = tk.Frame(main_frame, bg=self.bg)
        self.image_frame.pack(fill="x", padx=12, pady=(6, 0))
        self.image_preview_label = tk.Label(self.image_frame, bg=self.bg)
        self.image_preview_label.pack(side="left")
        self.clear_image_btn = tk.Button(
            self.image_frame, text="✕", command=self._clear_image,
            bg="#2a2a4a", fg="#ff4444", bd=0, padx=6, cursor="hand2",
        )

        # ─── Input Area ──────────────────────────────────────────────────────
        input_frame = tk.Frame(main_frame, bg=self.bg)
        input_frame.pack(fill="x", padx=12, pady=(8, 12))

        input_row = tk.Frame(input_frame, bg=self.bg)
        input_row.pack(fill="x")

        self.upload_btn = tk.Button(
            input_row, text="📷", command=self._upload_image,
            bg=self.card_bg, fg="#aaa", bd=1, relief="solid",
            highlightbackground=self.border, padx=12, pady=6,
            font=("Segoe UI", 16), cursor="hand2",
        )
        self.upload_btn.pack(side="left", padx=(0, 8))

        self.prompt_entry = tk.Text(
            input_row, height=2, font=("Segoe UI", 12),
            bg=self.card_bg, fg=self.fg, insertbackground=self.fg,
            bd=1, relief="solid", highlightbackground=self.border,
            padx=10, pady=8, wrap="word",
        )
        self.prompt_entry.pack(side="left", fill="x", expand=True)
        self.prompt_entry.bind("<Return>", self._on_enter)
        self.prompt_entry.bind("<Shift-Return>", lambda e: None)
        self.prompt_entry.focus()

        self.send_btn = tk.Button(
            input_row, text="Send →", command=self._send_message,
            bg=self.accent, fg="white", bd=0, padx=16, pady=6,
            font=("Segoe UI", 12, "bold"), cursor="hand2",
        )
        self.send_btn.pack(side="left", padx=(8, 0))

        # Reset button in header
        reset_btn = tk.Button(
            header, text="↻ Reset", command=self._reset_conversation,
            bg="#2a2a4a", fg="#aaa", bd=0, padx=10, pady=2,
            font=("Segoe UI", 10), cursor="hand2",
        )
        reset_btn.pack(side="right", padx=(0, 8), pady=10)

        # Welcome message
        self._add_message("assistant", "Hello! I'm Chitragupt, your vision-enabled agentic assistant. Upload an image and ask me anything, or just chat!")

    def _add_message(self, role: str, text: str, extras: dict | None = None):
        self.chat_display.configure(state="normal")
        if role == "user":
            self.chat_display.insert("end", f"\n🧑 You:\n", "user")
            self.chat_display.insert("end", f"{text}\n", "user")
        else:
            self.chat_display.insert("end", f"\n🤖 Chitragupt:\n", "assistant")
            self.chat_display.insert("end", f"{text}\n", "assistant")
            if extras:
                if extras.get("tool_calls"):
                    for tc in extras["tool_calls"]:
                        self.chat_display.insert("end", f"   ⚡ Used tool: {tc['tool']}\n", "tool")
                if extras.get("model"):
                    self.chat_display.insert("end", f"   ── {extras['model']}\n", "model")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp")],
        )
        if not path:
            return

        self.current_image_path = path
        with open(path, "rb") as f:
            self.current_image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Show preview
        pil_img = Image.open(path)
        pil_img.thumbnail((80, 80))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_preview_label.configure(image=tk_img)
        self.image_preview_label.image = tk_img
        self.clear_image_btn.pack(side="left", padx=(6, 0))
        self.image_frame.pack(fill="x", padx=12, pady=(6, 0))

    def _clear_image(self):
        self.current_image_base64 = None
        self.current_image_path = None
        self.image_preview_label.configure(image="")
        self.clear_image_btn.pack_forget()
        self.image_frame.pack_forget()

    def _on_enter(self, event):
        if not event.state & 0x1:  # Shift not pressed
            self._send_message()
            return "break"

    def _send_message(self):
        prompt = self.prompt_entry.get("1.0", "end-1c").strip()
        if not prompt and not self.current_image_base64:
            return

        self.prompt_entry.delete("1.0", "end")
        self.send_btn.configure(state="disabled", text="...")

        self._add_message("user", prompt or "(image uploaded)")

        # Run async in thread
        threading.Thread(target=self._do_chat, args=(prompt,), daemon=True).start()

    def _do_chat(self, prompt: str):
        import asyncio

        async def _request():
            resp = await self.api_client.post(
                f"{API_URL}/v1/chat",
                json={
                    "prompt": prompt or "What do you see in this image?",
                    "image_base64": self.current_image_base64,
                },
            )
            resp.raise_for_status()
            return resp.json()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_request())
            loop.close()

            self.root.after(0, self._handle_response, result)
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _handle_response(self, result: dict):
        self._add_message("assistant", result["text"], {
            "model": f"{result['provider']}/{result['model']}",
            "tool_calls": result.get("tool_calls", []),
        })
        self._clear_image()
        self.send_btn.configure(state="normal", text="Send →")

    def _handle_error(self, error: str):
        self._add_message("assistant", f"⚠️ Error: {error}")
        self.send_btn.configure(state="normal", text="Send →")

    def _reset_conversation(self):
        async def _reset():
            await self.api_client.post(f"{API_URL}/v1/reset")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_reset())
            loop.close()
        except:
            pass
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self._add_message("assistant", "Conversation reset. How can I help you?")

    def _check_health(self):
        async def _health():
            try:
                resp = await self.api_client.get(f"{API_URL}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    self.root.after(0, lambda: self.status_label.configure(
                        text=f"● Connected ({data.get('mode', '?')})", fg="#00ff88"
                    ))
                else:
                    self.root.after(0, lambda: self.status_label.configure(text="● Disconnected", fg="#ff4444"))
            except:
                self.root.after(0, lambda: self.status_label.configure(text="● Disconnected", fg="#ff4444"))

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_health())
            loop.close()
        except:
            pass

        # Retry every 10s
        self.root.after(10000, self._check_health)


def main():
    root = tk.Tk()
    app = ChitraguptDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
