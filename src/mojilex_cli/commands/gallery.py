"""Standalone local gallery of saved descriptions and retained preview frames."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import struct
import tempfile
import webbrowser
import zlib
from pathlib import Path
from typing import Any

from PIL import Image

from mojilex_cli.config import load_config
from mojilex_cli.media.resume import _png, _read

from .packs import resolve_pack_run, show_pack_command
from .runtime import CommandResult

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAX_PREVIEW_BYTES = 1024 * 1024
_MAX_GALLERY_PREVIEW_BYTES = 32 * 1024 * 1024
_WARNING_LABELS = {
    "nudity": "Нагота",
    "sexual-content": "Сексуальное содержание",
    "graphic-violence": "Сцены жестокости",
    "hate-symbol": "Символ ненависти",
    "self-harm": "Самоповреждение",
    "drugs": "Наркотики",
    "flashing": "Мигающие вспышки",
    "other-sensitive": "Другое чувствительное содержание",
}
_RATING_LABELS = {
    "general": "Общее",
    "sensitive": "Чувствительное",
    "adult": "Для взрослых",
    "unknown": "Не определено",  # noqa: RUF001
}
_SCRIPT = """const input = document.querySelector('#search');
const cards = [...document.querySelectorAll('article')];
const count = document.querySelector('#visible-count');
input.addEventListener('input', () => {
  const query = input.value.toLocaleLowerCase().trim();
  let visible = 0;
  for (const card of cards) {
    card.hidden = !card.textContent.toLocaleLowerCase().includes(query);
    if (!card.hidden) visible++;
  }
  count.textContent = String(visible);
});"""
_STYLE = """*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#172033;
font:16px/1.6 system-ui,sans-serif}header,main,footer{max-width:1120px;margin:auto;
padding:24px}header{padding-bottom:8px}h1{margin:0;font-size:32px}h2{font-size:18px;
margin:0 0 10px}p{margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.muted,footer{color:#596579}.toolbar{position:sticky;top:0;background:#f4f5f7;
padding:12px 0;z-index:1}label{display:block;font-weight:600}input{width:100%;
padding:12px 16px;border:1px solid #9aa5b4;border-radius:10px;font:inherit}
input:focus{outline:3px solid #adc9ff;border-color:#245bcc}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px}
article{background:white;border:1px solid #e0e5eb;border-radius:14px;padding:20px;
overflow:hidden}article[hidden]{display:none}.preview{width:128px;height:128px;
display:flex;align-items:center;justify-content:center;background:#eef1f5;
border-radius:12px;margin-bottom:14px}.preview img{width:128px;height:128px;
object-fit:contain}.preview.missing{font-size:12px;color:#596579;text-align:center;
padding:12px}.badge{display:inline-block;background:#fff1cc;color:#674500;
border-radius:6px;padding:3px 8px;margin:3px 4px 3px 0;font-size:14px}details{
border-top:1px solid #e0e5eb;margin-top:16px;padding-top:10px}summary{cursor:pointer;
color:#245bcc;font-weight:600}dl{margin:8px 0}dt{font-weight:600}dd{margin:0 0 8px;
overflow-wrap:anywhere}code{font-size:12px}ul{padding-left:24px}.notice{padding:12px;
background:#fff1cc;border-radius:8px}@media(max-width:600px){header,main,footer{
padding:16px}h1{font-size:26px}}"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _preview(root: Path, descriptor: str) -> bytes | None:
    """Read one checksum-verified generated PNG, never a path supplied by metadata."""
    if not _HASH.fullmatch(descriptor):
        return None
    try:
        entry = root / descriptor
        manifest = json.loads(_read(entry / "manifest.json", 32 * 1024))
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            return None
        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            return None
        frame = frames[0]
        if (
            not isinstance(frame, dict)
            or frame.get("name") != "light-00.png"
            or type(frame.get("bytes")) is not int
            or not 1 <= frame["bytes"] <= _MAX_PREVIEW_BYTES
        ):
            return None
        data = _read(entry / "light-00.png", frame["bytes"])
        if hashlib.sha256(data).hexdigest() != frame.get("sha256"):
            return None
        _png(data)
        return data
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        SyntaxError,
        struct.error,
        zlib.error,
        Image.DecompressionBombError,
    ):
        return None


def _previews(run_id: str, native_ids: set[str]) -> dict[str, bytes]:
    checkpoint = resolve_pack_run(run_id, purpose="view")
    config = load_config()
    if config.cache_dir is None:
        return {}
    root = config.cache_dir / "resume-media" / hashlib.sha256(run_id.encode()).hexdigest()
    result: dict[str, bytes] = {}
    total = 0
    for native_id in sorted(native_ids):
        element = checkpoint.elements.get(native_id)
        if element is None or element.source_descriptor_sha256 is None:
            continue
        data = _preview(root, element.source_descriptor_sha256)
        if data is not None and total + len(data) <= _MAX_GALLERY_PREVIEW_BYTES:
            result[native_id] = data
            total += len(data)
    return result


def _description(value: dict[str, Any]) -> str:
    parts = [f"<p>{_escape(value.get('text', 'Описание отсутствует'))}</p>"]
    if value.get("motion"):
        parts.append(f"<p><strong>Движение:</strong> {_escape(value['motion'])}</p>")
    if value.get("usage"):
        parts.append("<p><strong>Примеры использования:</strong></p><ul>")
        parts.extend(f"<li>{_escape(item)}</li>" for item in value["usage"])
        parts.append("</ul>")
    return "".join(parts)


def _card(item: dict[str, Any], number: int, preview: bytes | None) -> str:
    if preview is None:
        image = '<div class="preview missing">Локальное изображение не сохранено</div>'
    else:
        encoded = base64.b64encode(preview).decode("ascii")
        image = (
            '<div class="preview"><img alt="Сохранённый кадр эмодзи" '
            f'loading="lazy" src="data:image/png;base64,{encoded}"></div>'
        )
    descriptions = item.get("descriptions", {})
    content = item.get("content", {})
    warnings = "".join(
        '<span class="badge">'
        f"{_escape(_WARNING_LABELS.get(warning, warning))} "
        f"<code>({_escape(warning)})</code></span>"
        for warning in content.get("warnings", [])
    )
    english = descriptions.get("en")
    english_detail = (
        f"<details><summary>English description</summary>{_description(english)}</details>"
        if english
        else ""
    )
    tags = ", ".join(item.get("semantic_tags", [])) or "Нет"
    rating = content.get("rating", "unknown")
    details = [
        "<dt>Теги</dt><dd>" + _escape(tags) + "</dd>",
        "<dt>Содержание</dt><dd>"
        + _escape(_RATING_LABELS.get(rating, rating))
        + f" <code>({_escape(rating)})</code></dd>",
    ]
    for key, title in (
        ("text_content", "Текст на эмодзи"),
        ("content_types", "Типы содержания"),
        ("styles", "Стиль"),
        ("suggested_uses", "Варианты использования"),
        ("uncertainties", "Неуверенность анализа"),
    ):
        value = item.get("facets", {}).get(key)
        if value:
            details.append(
                f"<dt>{title}</dt><dd>" + _escape(json.dumps(value, ensure_ascii=False)) + "</dd>"
            )
    return (
        f"<article><h2>Эмодзи {number}</h2>{image}"
        + _description(descriptions.get("ru", {}))
        + warnings
        + english_detail
        + "<details><summary>Теги и подробности</summary><dl>"
        + "".join(details)
        + f"<dt>ID эмодзи</dt><dd>{_escape(item.get('native_id', ''))}</dd></dl></details>"
        + "</article>"
    )


def render_gallery(result: CommandResult, previews: dict[str, bytes]) -> str:
    """Render trusted markup around escaped saved data; no executable data payloads."""
    pack = result.result["pack"]
    items = result.result["items"]
    name = ", ".join(pack["names"])
    cards = "".join(
        _card(item, index, previews.get(item["native_id"])) for index, item in enumerate(items, 1)
    )
    script_hash = base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode("ascii")
    notices = "".join(f'<p class="notice">{_escape(warning)}</p>' for warning in result.warnings)
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
        "img-src data:; style-src &#39;unsafe-inline&#39;; "
        f"script-src &#39;sha256-{script_hash}&#39;; base-uri &#39;none&#39;; "
        'form-action &#39;none&#39;">'
        f"<title>{_escape(name)} · MojiLex</title><style>{_STYLE}</style></head><body>"
        f"<header><h1>{_escape(name)}</h1><p>Сохранённые описания: {len(items)} из "
        f'{int(pack["items"])}</p><p class="muted">Предупреждения — метки содержания. '
        "Ручное одобрение не требуется.</p>"
        f'{notices}<div class="toolbar"><label for="search">Поиск по описаниям и тегам</label>'
        '<input id="search" type="search" placeholder="Например: взрыв, радость, сердце" '
        'autocomplete="off"><p class="muted" aria-live="polite">Найдено: '
        f'<span id="visible-count">{len(items)}</span></p></div></header>'
        f'<main class="grid">{cards}</main><footer>Локальная копия результата. '
        "Страница работает без интернета и не изменяет сохранённый анализ. "
        f"Превью: {len(previews)} из {len(items)}. Для обновления откройте галерею снова."
        f"</footer><script>{_SCRIPT}</script></body></html>"
    )


def gallery_command(selector: str, *, open_browser: bool = True) -> CommandResult:
    """Create a local snapshot without invoking a provider or changing saved work."""
    saved = show_pack_command(selector)
    previews = (
        _previews(saved.run_id, {item["native_id"] for item in saved.result["items"]})
        if saved.run_id is not None
        else {}
    )
    document = render_gallery(saved, previews)
    descriptor, filename = tempfile.mkstemp(prefix="mojilex-gallery-", suffix=".html")
    path = Path(filename).absolute()
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    opened = False
    warnings = list(saved.warnings)
    if open_browser:
        try:
            opened = webbrowser.open(path.as_uri())
        except (OSError, webbrowser.Error):
            pass
        if not opened:
            warnings.append(
                "Не удалось открыть браузер. Откройте сохранённый HTML-файл."  # noqa: RUF001
            )
    return CommandResult(
        run_id=saved.run_id,
        result={
            "gallery_path": str(path),
            "browser_opened": opened,
            "pack": saved.result["pack"],
            "counts": saved.result["counts"],
            "preview_count": len(previews),
        },
        warnings=warnings,
    )
