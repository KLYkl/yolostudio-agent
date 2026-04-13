from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


DEFAULT_PAGE_URL = "https://www.skylinewebcams.com/zh/webcam/thailand/surat-thani/ko-samui/lamai.html"
DEFAULT_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
)


def _resolve_chrome_executable(explicit: str) -> str:
    candidates: list[str] = []
    if explicit.strip():
        candidates.append(explicit.strip())
    candidates.extend(DEFAULT_CHROME_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "未找到可用的 Chrome 可执行文件。可通过 --chrome-exe 显式指定。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a SkylineWebcams page, click play, wait for ads to finish, and capture the live HLS URL."
    )
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--wait-seconds", type=float, default=12.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--chrome-exe", default="")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    chrome_exe = _resolve_chrome_executable(args.chrome_exe)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None

    payload: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "page_url": args.page_url,
        "wait_seconds": float(args.wait_seconds),
        "chrome_exe": chrome_exe,
        "hls_url": "",
        "source": "",
        "video_current_src": "",
        "ok": False,
        "summary": "",
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=chrome_exe,
            headless=not args.headed,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        matched_hls: dict[str, str] = {"url": ""}

        def on_request(request: object) -> None:
            try:
                url = request.url  # type: ignore[attr-defined]
            except Exception:
                return
            if url.startswith("https://hd-auth.skylinewebcams.com/live.m3u8?a="):
                matched_hls["url"] = url

        page.on("request", on_request)

        try:
            page.goto(args.page_url, wait_until="domcontentloaded", timeout=120000)
            page.locator(".player-poster, video").first.click(force=True, timeout=30000)
            page.wait_for_timeout(max(int(args.wait_seconds * 1000), 1000))

            source = page.evaluate("() => window.player?.options?.source ?? ''")
            current_src = page.evaluate("() => document.querySelector('video')?.currentSrc ?? ''")
            payload["source"] = source
            payload["video_current_src"] = current_src

            if not matched_hls["url"] and isinstance(source, str) and "?a=" in source:
                matched_hls["url"] = "https://hd-auth.skylinewebcams.com/live.m3u8" + source[source.index("?") :]

            if not matched_hls["url"]:
                raise RuntimeError("未抓到 Skyline 真实 HLS 请求")

            payload["hls_url"] = matched_hls["url"]
            payload["ok"] = True
            payload["summary"] = "已抓到 Skyline 实时 HLS 地址，可继续做 RTSP 桥接"
        except PlaywrightTimeoutError as exc:
            payload["summary"] = f"抓取 Skyline HLS 超时: {exc}"
            raise
        finally:
            browser.close()

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload["output_json"] = str(output_json)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
