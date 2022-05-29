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

You may want to export the PATH of this plugin, e.g. to be used in python (replacing path_to_repo)
```
export GST_PLUGIN_PATH_1_0=/opt/homebrew/lib/gstreamer-1.0:/path_to_repo/gst-airplay-mac/build
```

Next you can test the plugin by calling 
```
gst-launch-1.0 airplaysrc ! queue ! h264parse ! decodebin ! videoconvert ! autovideosink
```

Use any iOS device on the same WiFi network and draw down the control center, where you can select the Screen Share icon. (You might need to go to the Settings > Control Center > More Controls section and tap + by the Screen Sharing option). The default pipeline may run a bit slowly, but don't worry in python everything should be fast.

To run a demo app, it is recommended to create a fresh pip environement (and deactivate current conda envs to avoid conflicts)
```
python3 -m venv pyair
source pyair/bin/activate 
pip install -r requirements.txt
cp gst_hacks.py pyair/lib/python3.9/site-packages/gstreamer/
```
The last line fixes a different naming convention of the python gstreamer lib so that it points to the homebrew folder and dylib file. 
Among others this installs streamlit, so the first thing you can try out is running:
```
streamlit run hello.py
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
