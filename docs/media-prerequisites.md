# Media prerequisites

WebP decoding is provided by Pillow and is installed with `mojilex-cli`. WebM requires `ffmpeg`
and `ffprobe`. TGS requires the small MojiLex lossless RGBA adapter from
`tools/rlottie_rgba_renderer.cpp`. `mojilex doctor` verifies an actual decode, not just the
presence of an executable.

On Windows, run `mojilex doctor` first. If FFmpeg or the TGS adapter is missing,
the interactive command offers to install the required components directly.
Answer `y`, or use `mojilex doctor --install`, instead of copying the commands
below. The TGS path may request administrator approval and install Visual Studio
Build Tools with the C++ workload before building the pinned adapter.

Typical FFmpeg installation commands are:

```console
# Windows
winget install --id Gyan.FFmpeg --exact

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

The upstream `lottie2gif` example flattens transparency and quantizes colors, so MojiLex rejects it.
Build the adapter against the exact CI-pinned Samsung rlottie revision below; do not substitute an
unreviewed binary download:

```console
git clone https://github.com/Samsung/rlottie.git
git -C rlottie checkout 683bbaa39dd0d366cf6b4bc300b4dfbee677ea6b
meson setup rlottie-build rlottie -Dexample=false -Dtest=false -Dmodule=false \
  -Ddefault_library=static --buildtype=release
meson compile -C rlottie-build
```

On Linux, compile and select the adapter (run from the `mojilex-cli` source checkout):

```console
c++ -std=c++14 -O2 -Wall -Wextra -Irlottie/inc tools/rlottie_rgba_renderer.cpp \
  "$PWD/rlottie-build/src/librlottie.a" -pthread \
  -o "$PWD/mojilex-rlottie-rgba"
export MOJILEX_RLOTTIE_RGBA="$PWD/mojilex-rlottie-rgba"
```

On macOS use the same command with Apple Clang. On Windows, run from a Visual Studio Developer
PowerShell after configuring rlottie as a static library:

```powershell
meson setup rlottie-build rlottie -Dexample=false -Dtest=false -Dmodule=false `
  -Ddefault_library=static --buildtype=release
meson compile -C rlottie-build
cl /std:c++14 /EHsc /O2 /I rlottie\inc tools\rlottie_rgba_renderer.cpp `
  /link /LIBPATH:rlottie-build\src rlottie.lib Shlwapi.lib /OUT:mojilex-rlottie-rgba.exe
$env:MOJILEX_RLOTTIE_RGBA = (Resolve-Path .\mojilex-rlottie-rgba.exe).Path
```

The adapter receives only already validated JSON plus the expected native width, height, and
`op-ip` frame count. It renders exactly that playable interval from rlottie's premultiplied
`0xAARRGGBB` surface, converts it deterministically to straight RGBA8 (transparent RGB is zero),
and writes an uncompressed, length-delimited private stream to stdout. The Python worker consumes
frames incrementally (there is no large raw temporary file) and rejects an invalid header,
dimensions, frame count, payload length, transparent RGB, truncation, or trailing bytes.

Run `mojilex doctor` after installation. Its generated TGS fixture contains a visible
semitransparent shape, so the check fails if alpha or the complete source timeline is lost.

Media is decoded in a child process with a clean environment, no shell, no network URL, and no API
credentials. If the current OS cannot enforce the required time and resource limits for a backend,
`doctor` reports that backend as unusable rather than claiming support.
