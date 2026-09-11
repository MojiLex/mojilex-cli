import base64
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from mojilex_cli.analysis import (
    AnalysisError,
    MediaAnalysisAccumulator,
    analyze_decoded_media,
    decoder_backend_fingerprint,
    known_profile_hashes,
    load_analysis_profile,
    sample_frame_indexes,
)
from mojilex_cli.analysis.engine import _dominant_palette
from mojilex_cli.analysis.models import RenderingSignals


def _image(width: int, height: int, rgba_hex: str) -> Image.Image:
    return Image.frombytes("RGBA", (width, height), bytes.fromhex(rgba_hex))


def _decoded_length(value: str) -> int:
    return len(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))


def test_bundled_profiles_are_exact_and_recursively_immutable() -> None:
    expected = {
        "color-v1": "fa7f0cb270bd3645b78ec5e6c1b8d6b23f8f2a07457699bc1ac95cf0c70aa230",
        "dedupe-v1": "c1f09fd2a4abb416b7dec82f67f9b101e1e9f993a8d578908a108f43d87602ba",
        "collection-dedupe-v1": (
            "5640836b227b4013772e6e48ce06e52245ad980bd0a554111d389504366fee8b"
        ),
    }
    assert dict(known_profile_hashes()) == expected
    for profile_id, expected_hash in expected.items():
        profile = load_analysis_profile(profile_id)
        assert profile.sha256 == expected_hash
        assert hashlib.sha256(profile.raw_bytes).hexdigest() == expected_hash
        assert profile.data["profile_id"] == profile_id
        assert "body" not in profile.data
        with pytest.raises(TypeError):
            profile.data["profile_id"] = "changed"  # type: ignore[index]

    color = load_analysis_profile("color-v1")
    assert isinstance(color.data["palette"], Mapping)
    with pytest.raises(TypeError):
        color.data["palette"]["max_colors"] = 99  # type: ignore[index]
    with pytest.raises(AnalysisError, match="unknown"):
        load_analysis_profile("dedupe-v2")


def test_decoder_backend_fingerprint_is_path_free_and_content_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_ffmpeg = tmp_path / "one" / "ffmpeg"
    second_ffmpeg = tmp_path / "two" / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    first_ffmpeg.parent.mkdir()
    second_ffmpeg.parent.mkdir()
    first_ffmpeg.write_bytes(b"same ffmpeg binary")
    second_ffmpeg.write_bytes(b"same ffmpeg binary")
    ffprobe.write_bytes(b"ffprobe binary")
    selected = {"ffmpeg": first_ffmpeg, "ffprobe": ffprobe}
    monkeypatch.setattr(
        "mojilex_cli.analysis.backend.shutil.which",
        lambda command: str(selected[command]),
    )
    first = decoder_backend_fingerprint("webm", webm_codec="vp9", webm_preserve_alpha=True)
    selected["ffmpeg"] = second_ffmpeg
    assert decoder_backend_fingerprint("webm", webm_codec="vp9", webm_preserve_alpha=True) == first
    second_ffmpeg.write_bytes(b"changed ffmpeg binary")
    assert decoder_backend_fingerprint("webm", webm_codec="vp9", webm_preserve_alpha=True) != first
    with pytest.raises(AnalysisError, match="requires codec"):
        decoder_backend_fingerprint("webm")
    with pytest.raises(AnalysisError, match="unknown"):
        decoder_backend_fingerprint("invalid")  # type: ignore[arg-type]


def test_tgs_backend_fingerprint_binds_the_lossless_rgba_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = tmp_path / "mojilex-rlottie-rgba"
    renderer.write_bytes(b"lossless adapter v1")
    monkeypatch.setattr(
        "mojilex_cli.analysis.backend.shutil.which",
        lambda _command: str(renderer),
    )
    first = decoder_backend_fingerprint("tgs", rlottie_renderer=str(renderer))
    renderer.write_bytes(b"lossless adapter changed")
    assert decoder_backend_fingerprint("tgs", rlottie_renderer=str(renderer)) != first


def test_dedupe_profile_normative_vectors() -> None:
    vectors = load_analysis_profile("dedupe-v1").data["normative_vectors"]
    assert isinstance(vectors, tuple)
    for vector in vectors:
        assert isinstance(vector, Mapping)
        input_data = vector["input"]
        expected = vector["expected"]
        assert isinstance(input_data, Mapping)
        assert isinstance(expected, Mapping)
        rgba_frames = input_data["rgba_hex"]
        durations = input_data["duration_us"]
        assert isinstance(rgba_frames, tuple)
        assert isinstance(durations, tuple)
        frames = [
            _image(int(input_data["width"]), int(input_data["height"]), str(rgba))
            for rgba in rgba_frames
        ]
        try:
            result = analyze_decoded_media(
                frames,
                durations_us=tuple(int(value) for value in durations),
                loop_mode=str(input_data["loop_mode"]),
            )
        finally:
            for frame in frames:
                frame.close()

        fingerprint = result.fingerprint
        perceptual = fingerprint.perceptual
        assert result.decoder_backend_fingerprint == decoder_backend_fingerprint("in-memory-rgba")
        assert fingerprint.decoded_payload_sha256 == expected["decoded_payload_sha256"]
        assert fingerprint.canonical_render_sha256 == expected["canonical_render_sha256"]
        assert fingerprint.shape_sha256 == expected["shape_sha256"]
        assert perceptual.layout_phash64 == expected["layout_phash64"]
        assert perceptual.content_phash64 == expected["content_phash64"]
        assert perceptual.alpha_phash64 == expected["alpha_phash64"]
        assert perceptual.edge_phash64 == expected["edge_phash64"]
        assert perceptual.temporal_energy_bp == expected["temporal_energy_bp"]
        assert perceptual.low_information is expected["low_information"]


def test_color_profile_normative_vectors() -> None:
    vectors = load_analysis_profile("color-v1").data["normative_vectors"]
    assert isinstance(vectors, tuple)
    for vector in vectors:
        assert isinstance(vector, Mapping)
        image = _image(int(vector["width"]), int(vector["height"]), str(vector["rgba_hex"]))
        try:
            result = analyze_decoded_media((image,))
        finally:
            image.close()
        rendering = result.rendering
        assert rendering.alpha_mode == vector["alpha_mode"]
        assert rendering.visible_area_bp == vector["visible_area_bp"]
        assert [color.model_dump() for color in rendering.dominant_colors or ()] == list(
            vector["dominant_colors"]
        )


def test_hidden_rgb_is_zeroed_before_all_exact_fingerprints() -> None:
    first = _image(2, 2, "ff0000ff112233001122330011223300")
    second = _image(2, 2, "ff0000ffffffff00abcdef0001020300")
    try:
        first_result = analyze_decoded_media((first,))
        second_result = analyze_decoded_media((second,))
    finally:
        first.close()
        second.close()
    assert first_result.fingerprint == second_result.fingerprint


def test_display_orientation_is_applied_before_native_stream_framing() -> None:
    source = Image.new("RGBA", (2, 3), (255, 0, 0, 255))
    source.putpixel((0, 0), (0, 255, 0, 255))
    source.getexif()[274] = 6  # rotate 90 degrees clockwise for display
    displayed = ImageOps.exif_transpose(source)
    try:
        oriented_result = analyze_decoded_media((source,))
        displayed_result = analyze_decoded_media((displayed,))
    finally:
        source.close()
        displayed.close()
    assert oriented_result.fingerprint == displayed_result.fingerprint


def test_native_alpha_mode_is_separate_from_canonical_padding() -> None:
    opaque = Image.new("RGBA", (4, 2), (0, 255, 0, 255))
    translucent = Image.new("RGBA", (4, 2), (0, 255, 0, 128))
    try:
        opaque_result = analyze_decoded_media((opaque,))
        translucent_result = analyze_decoded_media((translucent,))
    finally:
        opaque.close()
        translucent.close()
    assert opaque_result.rendering.alpha_mode == "opaque"
    assert opaque_result.rendering.visible_area_bp < 10_000
    assert translucent_result.rendering.alpha_mode == "translucent"
    assert translucent_result.rendering.visible_area_bp < opaque_result.rendering.visible_area_bp


def test_platform_adaptive_color_does_not_publish_service_rgb() -> None:
    image = Image.new("RGBA", (3, 3), (11, 22, 33, 255))
    try:
        result = analyze_decoded_media((image,), needs_repainting=True)
    finally:
        image.close()
    assert result.rendering.color_behavior == "platform-adaptive"
    assert result.rendering.dominant_colors is None


def test_visible_high_entropy_palette_always_has_publishable_integer_coverage() -> None:
    raw = bytearray()
    for red in range(0, 256, 8):
        for green in range(0, 256, 8):
            for blue in range(0, 256, 8):
                raw.extend((red, green, blue, 255))
    image = Image.frombytes("RGBA", (256, 128), bytes(raw))
    try:
        result = analyze_decoded_media((image,))
    finally:
        image.close()
    colors = result.rendering.dominant_colors
    assert colors is not None and len(colors) == 5
    assert all(color.coverage_bp == 1 for color in colors)


def test_palette_rounding_never_exceeds_ten_thousand_basis_points() -> None:
    image = Image.new("RGBA", (3, 2), (255, 0, 0, 255))
    image.putpixel((1, 1), (0, 255, 0, 255))
    image.putpixel((2, 1), (0, 0, 255, 255))
    try:
        colors, _ = _dominant_palette((image,))
    finally:
        image.close()
    assert sum(color.coverage_bp for color in colors) == 10_000
    assert sorted(color.coverage_bp for color in colors) == [1_667, 1_667, 6_666]


def test_analysis_model_rejects_palette_coverage_above_ten_thousand() -> None:
    with pytest.raises(ValueError, match="must not exceed 10000"):
        RenderingSignals(
            color_behavior="fixed",
            palette_dynamics="stable",
            alpha_mode="opaque",
            visible_area_bp=10_000,
            dominant_colors=(
                {"hex": "#ff0000", "family": "red", "coverage_bp": 6_667},
                {"hex": "#00ff00", "family": "green", "coverage_bp": 1_667},
                {"hex": "#0000ff", "family": "blue", "coverage_bp": 1_667},
            ),
        )


def test_animation_uses_sixteen_chronological_samples_and_detects_palette_change() -> None:
    frames = [
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)),
        Image.new("RGBA", (2, 2), (0, 0, 255, 255)),
    ]
    try:
        result = analyze_decoded_media(frames, durations_us=(500_000, 500_000))
    finally:
        for frame in frames:
            frame.close()
    perceptual = result.fingerprint.perceptual
    assert perceptual.sample_count == 16
    assert _decoded_length(perceptual.layout_phash64) == 16 * 8
    assert _decoded_length(perceptual.content_phash64) == 16 * 8
    assert _decoded_length(perceptual.alpha_phash64) == 16 * 8
    assert _decoded_length(perceptual.edge_phash64) == 16 * 8
    assert perceptual.temporal_energy_bp > 0
    assert result.rendering.palette_dynamics == "changing"
    assert [(item.family, item.coverage_bp) for item in result.rendering.dominant_colors or ()] == [
        ("blue", 5_000),
        ("red", 5_000),
    ]


def test_full_stream_hashes_include_frames_not_selected_for_perceptual_sampling() -> None:
    durations = (1,) * 17
    sampled = set(sample_frame_indexes(durations, 16))
    omitted = next(index for index in range(17) if index not in sampled)
    baseline = [Image.new("RGBA", (2, 2), (255, 0, 0, 255)) for _ in durations]
    changed = [frame.copy() for frame in baseline]
    changed[omitted].putpixel((0, 0), (0, 255, 0, 255))
    try:
        baseline_result = analyze_decoded_media(baseline, durations_us=durations)
        changed_result = analyze_decoded_media(changed, durations_us=durations)
    finally:
        for frame in (*baseline, *changed):
            frame.close()
    assert baseline_result.fingerprint.perceptual == changed_result.fingerprint.perceptual
    assert (
        baseline_result.fingerprint.decoded_payload_sha256
        != changed_result.fingerprint.decoded_payload_sha256
    )
    assert (
        baseline_result.fingerprint.canonical_render_sha256
        != changed_result.fingerprint.canonical_render_sha256
    )
    # Shape is intentionally color-independent.
    assert baseline_result.fingerprint.shape_sha256 == changed_result.fingerprint.shape_sha256


def test_accumulator_fails_closed_for_incomplete_or_inconsistent_streams() -> None:
    image = Image.new("RGBA", (2, 2), "red")
    try:
        accumulator = MediaAnalysisAccumulator(
            width=2,
            height=2,
            frame_count=2,
            animated=True,
            loop_mode="loop",
            needs_repainting=False,
        )
        accumulator.add_frame(image, duration_us=1)
        with pytest.raises(AnalysisError, match="ended before"):
            accumulator.finalize((image,) * 16)

        wrong_size = MediaAnalysisAccumulator(
            width=1,
            height=1,
            frame_count=1,
            animated=False,
            loop_mode="once",
            needs_repainting=False,
        )
        with pytest.raises(AnalysisError, match="dimensions"):
            wrong_size.add_frame(image, duration_us=0)
    finally:
        image.close()

    with pytest.raises(AnalysisError, match="unsafe"):
        MediaAnalysisAccumulator(
            width=20_000,
            height=20_000,
            frame_count=1,
            animated=False,
            loop_mode="once",
            needs_repainting=False,
        )

    static = MediaAnalysisAccumulator(
        width=1,
        height=1,
        frame_count=1,
        animated=False,
        loop_mode="once",
        needs_repainting=False,
    )
    static_frame = Image.new("RGBA", (1, 1), "red")
    try:
        with pytest.raises(AnalysisError, match="duration"):
            static.add_frame(static_frame, duration_us=1)
    finally:
        static_frame.close()
