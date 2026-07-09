"""Chitragupt CLI — interact with the agent from the terminal."""

from __future__ import annotations
import argparse
import asyncio
import base64
import sys
from pathlib import Path

import httpx


class ChitraguptClient:
    """Client for the Chitragupt API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, prompt: str, image_path: str | None = None) -> dict:
        """Send a chat request with optional image."""
        image_base64 = None
        if image_path:
            path = Path(image_path)
            if not path.exists():
                return {"error": f"File not found: {image_path}"}
            with open(path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

        resp = await self.client.post(
            f"{self.base_url}/v1/chat",
            json={"prompt": prompt, "image_base64": image_base64},
        )
        resp.raise_for_status()
        return resp.json()

    async def reset(self):
        resp = await self.client.post(f"{self.base_url}/v1/reset")
        return resp.json()

    async def close(self):
        await self.client.aclose()


async def interactive_mode(client: ChitraguptClient):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("  ✦ Chitragupt — Agentic Vision Assistant")
    print("  Commands: /reset, /image <path>, /quit")
    print("=" * 60)

    current_image = None

    while True:
        try:
            text = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text == "/quit":
            break
        elif text == "/reset":
            await client.reset()
            print("🔄 Conversation reset.")
            current_image = None
            continue
        elif text.startswith("/image "):
            current_image = text.split(" ", 1)[1].strip()
            print(f"📷 Image set: {current_image}")
            continue

        print("\n🤖 Chitragupt is thinking...", end="", flush=True)
        try:
            result = await client.chat(text, current_image)
            print("\r", end="")
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                print(f"\n🤖 {result['text']}")
                if result.get("tool_calls"):
                    for tc in result["tool_calls"]:
                        print(f"   ⚡ Used tool: {tc['tool']}")
                print(f"   ── ({result['provider']}/{result['model']})")
        except Exception as e:
            print(f"\r❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chitragupt CLI — Vision-based Agentic Assistant")
    parser.add_argument("prompt", nargs="?", help="Prompt to send")
    parser.add_argument("--image", "-i", help="Path to an image file")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--interactive", "-I", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    client = ChitraguptClient(args.url)

    if args.interactive or not args.prompt:
        asyncio.run(interactive_mode(client))
    else:
        result = asyncio.run(client.chat(args.prompt, args.image))
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(result["text"])
            if result.get("tool_calls"):
                for tc in result["tool_calls"]:
                    print(f"   ⚡ Used tool: {tc['tool']}")


if __name__ == "__main__":
    main()
