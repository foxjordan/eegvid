# BioSemi driver library goes here

Download the RingBuffer SDK for your OS from
https://www.biosemi.com/faq/make_own_acquisition_software.htm
and copy the library into this folder before building:

| OS      | file                                   |
|---------|----------------------------------------|
| Windows | `RingBuffer.dll` (or `Labview_DLL.dll`) |
| macOS   | `BioSemiRingBuffer64.framework/` (from the SDK's `OSX64`, or inside ActiView-Mac's `Contents/Resources`); Intel-only, so build the app with x86_64 Python |
| Linux   | `libRingBuffer.so`                     |

The build bundles it into the app. Check BioSemi's terms before
distributing the library; it is not committed to git (see `.gitignore`).

Also install the BioSemi USB driver itself (Windows: from the ActiView
download) on each acquisition PC.
