"""Versioned, deterministic, network-free color and fingerprint analysis."""

from __future__ import annotations

import base64
import hashlib
import math
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from itertools import pairwise

from PIL import Image, ImageOps

from .backend import decoder_backend_fingerprint
from .models import (
    AnalysisError,
    DeterministicMediaAnalysis,
    DominantColor,
    FingerprintSignals,
    PerceptualSignals,
    RenderingSignals,
)
from .profiles import load_analysis_profile

_CANVAS = 256
_CONTENT_BOX = 224
_ALPHA_THRESHOLD = 16
_LOOP_CODES = {"once": 0, "loop": 1, "unknown": 255}
_PHASH_SIZE = 32
_PHASH_LOW = 8
_COSINE_SCALE = 1 << 14


class MediaAnalysisAccumulator:
    """Stream full decoded frames into exact hashes while retaining no media."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_count: int,
        animated: bool,
        loop_mode: str,
        needs_repainting: bool,
        backend_fingerprint: str | None = None,
    ) -> None:
        limits = load_analysis_profile("dedupe-v1").data.get("resource_limits")
        if not isinstance(limits, Mapping):
            raise AnalysisError("dedupe analysis resource limits are invalid")
        maximum_frames = limits.get("max_full_frames")
        maximum_bytes = limits.get("max_decoded_rgba_bytes")
        if not isinstance(maximum_frames, int) or not isinstance(maximum_bytes, int):
            raise AnalysisError("dedupe analysis resource limits are incomplete")
        if (
            width <= 0
            or height <= 0
            or frame_count <= 0
            or frame_count > maximum_frames
            or width * height * 4 * frame_count > maximum_bytes
        ):
            raise AnalysisError("decoded stream dimensions or frame count are unsafe")
        if loop_mode not in _LOOP_CODES:
            raise AnalysisError("decoded stream loop mode is invalid")
        if not animated and frame_count != 1:
            raise AnalysisError("decoded stream animation flag and frame count disagree")
        if not animated and loop_mode != "once":
            raise AnalysisError("a static decoded stream must use once loop mode")
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.animated = animated
        self.loop_mode = loop_mode
        self.needs_repainting = needs_repainting
        self.backend_fingerprint = backend_fingerprint or decoder_backend_fingerprint(
            "in-memory-rgba"
        )
        if len(self.backend_fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in self.backend_fingerprint
        ):
            raise AnalysisError("decoder backend fingerprint is invalid")
        self._frames_seen = 0
        self._has_zero_alpha = False
        self._has_intermediate_alpha = False

        loop_code = _LOOP_CODES[loop_mode]
        self._decoded = hashlib.sha256()
        self._decoded.update(b"MLXDP1\0")
        self._decoded.update(struct.pack(">IIIB", width, height, frame_count, loop_code))
        self._canonical = hashlib.sha256()
        self._canonical.update(b"MLXCR1\0")
        self._canonical.update(struct.pack(">IB", frame_count, loop_code))
        self._shape = hashlib.sha256()
        self._shape.update(b"MLXSH1\0")
        self._shape.update(struct.pack(">IB", frame_count, loop_code))

    def add_frame(self, image: Image.Image, *, duration_us: int) -> None:
        if self._frames_seen >= self.frame_count:
            raise AnalysisError("decoded stream contains more frames than declared")
        if (
            duration_us < 0
            or (self.animated and duration_us == 0)
            or (not self.animated and duration_us != 0)
        ):
            raise AnalysisError("decoded frame presentation duration is invalid")
        rgba = _normalized_native_rgba(image)
        try:
            if rgba.size != (self.width, self.height):
                raise AnalysisError("decoded stream changed native canvas dimensions")
            alpha_histogram = rgba.getchannel("A").histogram()
            self._has_zero_alpha = self._has_zero_alpha or alpha_histogram[0] > 0
            self._has_intermediate_alpha = self._has_intermediate_alpha or any(
                alpha_histogram[1:255]
            )
            native = rgba.tobytes()
            self._decoded.update(struct.pack(">QQ", duration_us, len(native)))
            self._decoded.update(native)

            layout = _layout_canvas(rgba)
            try:
                layout_bytes = layout.tobytes()
                self._canonical.update(struct.pack(">Q", duration_us))
                self._canonical.update(layout_bytes)
                alpha = layout.getchannel("A")
                binary_alpha = alpha.point(
                    tuple(255 if value >= _ALPHA_THRESHOLD else 0 for value in range(256))
                )
                self._shape.update(struct.pack(">Q", duration_us))
                self._shape.update(binary_alpha.tobytes())
                binary_alpha.close()
                alpha.close()
            finally:
                layout.close()
            self._frames_seen += 1
        finally:
            rgba.close()

    def finalize(self, perceptual_frames: Sequence[Image.Image]) -> DeterministicMediaAnalysis:
        if self._frames_seen != self.frame_count:
            raise AnalysisError("decoded stream ended before its declared frame count")
        expected_samples = 16 if self.animated else 1
        if len(perceptual_frames) != expected_samples:
            raise AnalysisError(
                f"perceptual analysis requires exactly {expected_samples} temporal samples"
            )
        samples = [_normalized_native_rgba(frame) for frame in perceptual_frames]
        try:
            if any(frame.size != (self.width, self.height) for frame in samples):
                raise AnalysisError("perceptual sample does not use the native canvas")
            rendering = _rendering_signals(
                samples,
                animated=self.animated,
                needs_repainting=self.needs_repainting,
                has_zero_alpha=self._has_zero_alpha,
                has_intermediate_alpha=self._has_intermediate_alpha,
            )
            perceptual = _perceptual_signals(
                samples,
                animated=self.animated,
                loop_mode=self.loop_mode,
                visible_area_bp=rendering.visible_area_bp,
            )
        finally:
            for frame in samples:
                frame.close()
        color_profile = load_analysis_profile("color-v1")
        dedupe_profile = load_analysis_profile("dedupe-v1")
        return DeterministicMediaAnalysis(
            color_profile_sha256=color_profile.sha256,
            dedupe_profile_sha256=dedupe_profile.sha256,
            decoder_backend_fingerprint=self.backend_fingerprint,
            rendering=rendering,
            fingerprint=FingerprintSignals(
                decoded_payload_sha256=self._decoded.hexdigest(),
                canonical_render_sha256=self._canonical.hexdigest(),
                shape_sha256=self._shape.hexdigest(),
                perceptual=perceptual,
            ),
        )


def analyze_decoded_media(
    frames: Sequence[Image.Image],
    *,
    durations_us: Sequence[int] | None = None,
    loop_mode: str | None = None,
    needs_repainting: bool = False,
    backend_fingerprint: str | None = None,
) -> DeterministicMediaAnalysis:
    """Analyze a complete in-memory decoded stream (primarily for static media/tests)."""

    if not frames:
        raise AnalysisError("decoded analysis requires at least one frame")
    animated = len(frames) > 1
    durations = tuple(durations_us or ((0,) if not animated else ()))
    if len(durations) != len(frames):
        raise AnalysisError("one presentation duration is required per decoded frame")
    selected_loop_mode = loop_mode or ("loop" if animated else "once")
    normalized_first = _normalized_native_rgba(frames[0])
    try:
        native_width, native_height = normalized_first.size
    finally:
        normalized_first.close()
    accumulator = MediaAnalysisAccumulator(
        width=native_width,
        height=native_height,
        frame_count=len(frames),
        animated=animated,
        loop_mode=selected_loop_mode,
        needs_repainting=needs_repainting,
        backend_fingerprint=backend_fingerprint,
    )
    for frame, duration_us in zip(frames, durations, strict=True):
        accumulator.add_frame(frame, duration_us=duration_us)
    sample_count = 16 if animated else 1
    indexes = sample_frame_indexes(durations, sample_count)
    return accumulator.finalize(tuple(frames[index] for index in indexes))


def sample_frame_indexes(durations_us: Sequence[int], sample_count: int) -> tuple[int, ...]:
    """Map exact normalized midpoint times to fully decoded presentation intervals."""

    if not durations_us or sample_count <= 0 or any(value < 0 for value in durations_us):
        raise AnalysisError("frame durations and sample count must be positive")
    if len(durations_us) == 1 and durations_us[0] == 0:
        return (0,) * sample_count
    duration = sum(durations_us)
    if duration <= 0:
        raise AnalysisError("animated decoded duration must be positive")
    result: list[int] = []
    for index in range(sample_count):
        # Compare integer rationals: target=(2i+1)*D/(2N), avoiding float drift.
        target_numerator = (2 * index + 1) * duration
        elapsed = 0
        selected = len(durations_us) - 1
        for frame_index, frame_duration in enumerate(durations_us):
            elapsed += frame_duration
            if target_numerator < 2 * sample_count * elapsed:
                selected = frame_index
                break
        result.append(selected)
    return tuple(result)


def _normalized_native_rgba(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    rgba = oriented.convert("RGBA")
    if oriented is not image:
        oriented.close()
    raw = bytearray(rgba.tobytes())
    for offset in range(3, len(raw), 4):
        if raw[offset] == 0:
            raw[offset - 3 : offset] = b"\0\0\0"
    normalized = Image.frombytes("RGBA", rgba.size, bytes(raw))
    rgba.close()
    return normalized


def _layout_canvas(image: Image.Image) -> Image.Image:
    resized = _fit_rgba(image, _CONTENT_BOX)
    canvas = Image.new("RGBA", (_CANVAS, _CANVAS), (0, 0, 0, 0))
    offset = ((_CANVAS - resized.width) // 2, (_CANVAS - resized.height) // 2)
    canvas.paste(resized, offset)
    resized.close()
    return canvas


def _content_canvas(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    visible = alpha.point(tuple(255 if value >= _ALPHA_THRESHOLD else 0 for value in range(256)))
    box = visible.getbbox()
    visible.close()
    alpha.close()
    if box is None:
        return Image.new("RGBA", (_CANVAS, _CANVAS), (0, 0, 0, 0))
    left, top, right, bottom = box
    margin_x = (max(1, right - left) * 5 + 99) // 100
    margin_y = (max(1, bottom - top) * 5 + 99) // 100
    box = (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    )
    cropped = image.crop(box)
    try:
        return _layout_canvas(cropped)
    finally:
        cropped.close()


def _fit_rgba(image: Image.Image, box_size: int) -> Image.Image:
    width, height = image.size
    if width >= height:
        target_width = box_size
        target_height = max(1, (height * box_size + width // 2) // width)
    else:
        target_height = box_size
        target_width = max(1, (width * box_size + height // 2) // height)
    source = image.load()
    target = Image.new("RGBA", (target_width, target_height))
    destination = target.load()
    if source is None or destination is None:
        raise AnalysisError("Pillow did not expose deterministic pixel access")
    for y in range(target_height):
        source_y = min(height - 1, ((2 * y + 1) * height) // (2 * target_height))
        for x in range(target_width):
            source_x = min(width - 1, ((2 * x + 1) * width) // (2 * target_width))
            destination[x, y] = source[source_x, source_y]
    return target


def _rendering_signals(
    frames: Sequence[Image.Image],
    *,
    animated: bool,
    needs_repainting: bool,
    has_zero_alpha: bool,
    has_intermediate_alpha: bool,
) -> RenderingSignals:
    if has_intermediate_alpha:
        alpha_mode = "translucent"
    elif has_zero_alpha:
        alpha_mode = "binary"
    else:
        alpha_mode = "opaque"
    layouts = [_layout_canvas(frame) for frame in frames]
    try:
        alpha_total = 0
        for layout in layouts:
            alpha = layout.getchannel("A")
            try:
                alpha_total += sum(alpha.tobytes())
            finally:
                alpha.close()
    finally:
        for layout in layouts:
            layout.close()
    denominator = len(frames) * _CANVAS * _CANVAS * 255
    visible_area_bp = _round_ratio(alpha_total * 10_000, denominator)
    palette, family_frames = _dominant_palette(frames)
    dynamics = "changing" if animated and _palette_changes(family_frames) else "stable"
    behavior = "platform-adaptive" if needs_repainting else "fixed"
    return RenderingSignals(
        color_behavior=behavior,
        palette_dynamics=dynamics,
        alpha_mode=alpha_mode,
        visible_area_bp=visible_area_bp,
        dominant_colors=None if needs_repainting or visible_area_bp == 0 else palette,
    )


def _dominant_palette(
    frames: Sequence[Image.Image],
) -> tuple[tuple[DominantColor, ...], tuple[dict[str, int], ...]]:
    scores: defaultdict[tuple[int, int, int], Fraction] = defaultdict(Fraction)
    channel_sums: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    family_frames: list[dict[str, int]] = []
    for frame in frames:
        bins: defaultdict[tuple[int, int, int], int] = defaultdict(int)
        family_weights: defaultdict[str, int] = defaultdict(int)
        total = 0
        raw = frame.tobytes()
        for offset in range(0, len(raw), 4):
            red, green, blue, alpha = raw[offset : offset + 4]
            if alpha < _ALPHA_THRESHOLD:
                continue
            key = (red >> 3, green >> 3, blue >> 3)
            bins[key] += alpha
            values = channel_sums[key]
            values[0] += red * alpha
            values[1] += green * alpha
            values[2] += blue * alpha
            values[3] += alpha
            family_weights[_color_family(red, green, blue)] += alpha
            total += alpha
        if total:
            for key, weight in bins.items():
                scores[key] += Fraction(weight, total)
            family_frames.append(
                {
                    family: _round_ratio(weight * 10_000, total)
                    for family, weight in family_weights.items()
                }
            )
        else:
            family_frames.append({})
    if not scores:
        return (), tuple(family_frames)
    selected = sorted(scores, key=lambda key: (-scores[key], key))[:5]
    coverage_by_key = _apportion_palette_coverage(selected, scores, len(frames))
    colors: list[DominantColor] = []
    for key in selected:
        red_sum, green_sum, blue_sum, weight = channel_sums[key]
        red = _round_ratio(red_sum, weight)
        green = _round_ratio(green_sum, weight)
        blue = _round_ratio(blue_sum, weight)
        colors.append(
            DominantColor(
                hex=f"#{red:02x}{green:02x}{blue:02x}",
                family=_color_family(red, green, blue),
                coverage_bp=coverage_by_key[key],
            )
        )
    colors.sort(key=lambda color: (-color.coverage_bp, color.family, color.hex))
    return tuple(colors), tuple(family_frames)


def _apportion_palette_coverage(
    selected: Sequence[tuple[int, int, int]],
    scores: Mapping[tuple[int, int, int], Fraction],
    frame_count: int,
) -> dict[tuple[int, int, int], int]:
    """Round selected cluster shares without ever exceeding 10,000 bp in total."""

    raw = {key: scores[key] * Fraction(10_000, frame_count) for key in selected}
    raw_total = sum(raw.values(), start=Fraction())
    target = min(10_000, max(len(selected), _round_fraction(raw_total)))
    coverage = {key: max(1, value.numerator // value.denominator) for key, value in raw.items()}

    current = sum(coverage.values())
    if current < target:
        order = sorted(
            selected,
            key=lambda key: (-(raw[key] - int(raw[key])), key),
        )
        for key in order[: target - current]:
            coverage[key] += 1
    elif current > target:
        order = sorted(
            (key for key in selected if coverage[key] > 1),
            key=lambda key: (raw[key] - int(raw[key]), key),
        )
        remaining = current - target
        while remaining:
            changed = False
            for key in order:
                if coverage[key] <= 1:
                    continue
                coverage[key] -= 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
            if not changed:  # Defensive: target is always at least len(selected).
                raise AnalysisError("palette coverage apportionment could not reach its target")
    return coverage


def _palette_changes(frames: Sequence[dict[str, int]]) -> bool:
    for left_index, left in enumerate(frames):
        for right in frames[left_index + 1 :]:
            families = set(left) | set(right)
            if sum(abs(left.get(name, 0) - right.get(name, 0)) for name in families) >= 2500:
                return True
            if any(
                max(left.get(name, 0), right.get(name, 0)) >= 1500
                and min(left.get(name, 0), right.get(name, 0)) < 250
                for name in families
            ):
                return True
    return False


def _color_family(red: int, green: int, blue: int) -> str:
    maximum, minimum = max(red, green, blue), min(red, green, blue)
    delta = maximum - minimum
    value_bp = _round_ratio(maximum * 10_000, 255)
    saturation_bp = 0 if maximum == 0 else _round_ratio(delta * 10_000, maximum)
    if value_bp <= 1500:
        return "black"
    if saturation_bp <= 1200:
        return "white" if value_bp >= 9000 else "gray"
    if maximum == red:
        hue_numerator = 60 * (green - blue)
        if hue_numerator < 0:
            hue_numerator += 360 * delta
    elif maximum == green:
        hue_numerator = 60 * (blue - red) + 120 * delta
    else:
        hue_numerator = 60 * (red - green) + 240 * delta
    if 15 * delta <= hue_numerator < 55 * delta and value_bp < 6500 and saturation_bp > 2500:
        return "brown"
    if (
        25 * delta <= hue_numerator < 60 * delta
        and value_bp >= 6500
        and 1000 < saturation_bp <= 4500
    ):
        return "beige"
    if hue_numerator < 15 * delta or hue_numerator >= 345 * delta:
        return "red"
    if hue_numerator < 45 * delta:
        return "orange"
    if hue_numerator < 70 * delta:
        return "yellow"
    if hue_numerator < 165 * delta:
        return "green"
    if hue_numerator < 200 * delta:
        return "cyan"
    if hue_numerator < 255 * delta:
        return "blue"
    if hue_numerator < 290 * delta:
        return "purple"
    return "pink"


def _perceptual_signals(
    frames: Sequence[Image.Image],
    *,
    animated: bool,
    loop_mode: str,
    visible_area_bp: int,
) -> PerceptualSignals:
    layout_hashes: list[int] = []
    content_hashes: list[int] = []
    alpha_hashes: list[int] = []
    edge_hashes: list[int] = []
    layout_luminances: list[bytes] = []
    entropy_histogram = [0] * 16
    edge_pixels = 0
    for frame in frames:
        layout = _layout_canvas(frame)
        content = _content_canvas(frame)
        try:
            layout_luma = _premultiplied_luminance(layout)
            content_luma = _premultiplied_luminance(content)
            alpha = layout.getchannel("A")
            edge = _edge_map(layout_luma, layout.size)
            layout_hashes.append(_phash64(layout_luma, layout.size))
            content_hashes.append(_phash64(content_luma, content.size))
            alpha_hashes.append(_phash64(alpha.tobytes(), alpha.size))
            edge_hashes.append(_phash64(edge, layout.size))
            layout_luminances.append(layout_luma)
            alpha_bytes = alpha.tobytes()
            for pixel_index, alpha_value in enumerate(alpha_bytes):
                if alpha_value >= _ALPHA_THRESHOLD:
                    raw_luma = min(255, layout_luma[pixel_index] * 255 // alpha_value)
                    entropy_histogram[raw_luma >> 4] += alpha_value
            edge_pixels += sum(value >= 32 for value in edge)
            alpha.close()
        finally:
            layout.close()
            content.close()
    temporal_energy = _temporal_energy(layout_luminances, loop_mode)
    entropy = _entropy_millibits(entropy_histogram)
    edge_density = _round_ratio(edge_pixels * 10_000, len(frames) * _CANVAS * _CANVAS)
    low_information = visible_area_bp <= 150 or entropy <= 250 or edge_density <= 50
    return PerceptualSignals(
        sample_count=16 if animated else 1,
        layout_phash64=_pack_hashes(layout_hashes),
        content_phash64=_pack_hashes(content_hashes),
        alpha_phash64=_pack_hashes(alpha_hashes),
        edge_phash64=_pack_hashes(edge_hashes),
        temporal_energy_bp=temporal_energy,
        low_information=low_information,
    )


def _premultiplied_luminance(image: Image.Image) -> bytes:
    values = bytearray(image.width * image.height)
    raw = image.tobytes()
    for index, offset in enumerate(range(0, len(raw), 4)):
        red, green, blue, alpha = raw[offset : offset + 4]
        luminance = (299 * red + 587 * green + 114 * blue + 500) // 1000
        values[index] = (luminance * alpha + 127) // 255
    return bytes(values)


def _edge_map(luminance: bytes, size: tuple[int, int]) -> bytes:
    width, height = size
    result = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = y * width + x
            horizontal = abs(luminance[index] - luminance[index - 1]) if x else 0
            vertical = abs(luminance[index] - luminance[index - width]) if y else 0
            result[index] = min(255, horizontal + vertical)
    return bytes(result)


def _phash64(values: bytes, size: tuple[int, int]) -> int:
    resized = _resize_l_nearest(values, size, (_PHASH_SIZE, _PHASH_SIZE))
    coefficients: list[int] = []
    for vertical in range(_PHASH_LOW):
        vertical_cosines = _COSINE_Q14[vertical]
        for horizontal in range(_PHASH_LOW):
            horizontal_cosines = _COSINE_Q14[horizontal]
            coefficient = 0
            for y in range(_PHASH_SIZE):
                row = y * _PHASH_SIZE
                row_sum = sum(resized[row + x] * horizontal_cosines[x] for x in range(_PHASH_SIZE))
                coefficient += row_sum * vertical_cosines[y]
            coefficients.append(coefficient)
    median = sorted(coefficients[1:])[(len(coefficients) - 1) // 2]
    result = 0
    for coefficient in coefficients:
        result = (result << 1) | int(coefficient > median)
    return result


def _resize_l_nearest(
    values: bytes, source_size: tuple[int, int], target_size: tuple[int, int]
) -> bytes:
    source_width, source_height = source_size
    target_width, target_height = target_size
    output = bytearray(target_width * target_height)
    for y in range(target_height):
        source_y = min(source_height - 1, ((2 * y + 1) * source_height) // (2 * target_height))
        for x in range(target_width):
            source_x = min(source_width - 1, ((2 * x + 1) * source_width) // (2 * target_width))
            output[y * target_width + x] = values[source_y * source_width + source_x]
    return bytes(output)


def _temporal_energy(frames: Sequence[bytes], loop_mode: str) -> int:
    if len(frames) <= 1:
        return 0
    pairs = list(pairwise(frames))
    if loop_mode == "loop":
        pairs.append((frames[-1], frames[0]))
    difference = sum(
        abs(left_value - right_value)
        for left, right in pairs
        for left_value, right_value in zip(left, right, strict=True)
    )
    return _round_ratio(difference * 10_000, len(pairs) * len(frames[0]) * 255)


def _entropy_millibits(histogram: Sequence[int]) -> int:
    total = sum(histogram)
    if total == 0:
        return 0
    with localcontext() as context:
        context.prec = 50
        decimal_total = Decimal(total)
        log_two = Decimal(2).ln()
        entropy = -sum(
            (
                (probability := Decimal(value) / decimal_total) * (probability.ln() / log_two)
                for value in histogram
                if value
            ),
            start=Decimal(0),
        )
        return int((entropy * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _pack_hashes(values: Sequence[int]) -> str:
    payload = b"".join(struct.pack(">Q", value) for value in values)
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0 or numerator < 0:
        raise AnalysisError("cannot round an invalid non-negative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def _round_fraction(value: Fraction) -> int:
    return _round_ratio(value.numerator, value.denominator)


def _cosine_q14(frequency: int, position: int) -> int:
    value = math.cos((2 * position + 1) * frequency * math.pi / (2 * _PHASH_SIZE))
    scaled = value * _COSINE_SCALE
    return int(math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5))


_COSINE_Q14 = tuple(
    tuple(_cosine_q14(frequency, position) for position in range(_PHASH_SIZE))
    for frequency in range(_PHASH_LOW)
)
