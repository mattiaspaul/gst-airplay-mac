A GStreamer plugin that provides an `airplaysrc` element for receiving video
streamed from Apple devices using the AirPlay protocol. Audio is currently not
supported.

## Edit by mattiaspaul

minor modifications to enable the usage of this gstreamer plugin on MacOS. In contrast to UxPlay this enables a more flexible integration as python plugin, e.g. to run deep networks on screenshare videos in Streamlit.

## Installation

Prerequirements are homebrew installed gstreamer libraries:
gstreamer1.0-tools, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, gstreamer1.0-libav, libgstreamer1.0-dev
Along with basic tools: cmake, meson, ninja

You can simply run the following
```
cd gst-airplay-mac
meson build
ninja -C meson
```

Afterwards you can inspect whether the dynamic library plugin is correctly built:
```
gst-inspect-1.0 build/libgstairplay.dylib
```


## Credits


The pluging code is based on
[gst-airplay](https://github.com/knuesel/gst-airplay) with minor modifications and
[gst-template](https://gitlab.freedesktop.org/gstreamer/gst-template/) and uses
the AirPlay implementation from [RPiPlay](https://github.com/FD-/RPiPlay) with
some changes (published in [this fork](https://github.com/knuesel/RPiPlay)) and additional changes from
[UxPlay fork](https://github.com/FDH2/UxPlay)

## License

The plugin makes use of [RPiPlay](https://github.com/FD-/RPiPlay) so it is
licensed under the GNU GPL 3.0.
