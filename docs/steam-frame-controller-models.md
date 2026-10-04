# Steam Frame controller models

WiVRn can generate its Steam Frame visual controller profile from the render-model files in a local SteamVR installation. The source assets are not included in this repository.

Enable the profile when configuring the Linux client:

```sh
cmake -S . -B build \
  -DWIVRN_BUILD_CLIENT=ON \
  -DWIVRN_STEAM_FRAME_CONTROLLER_PROFILE=ON \
  -DWIVRN_STEAM_FRAME_RENDERMODELS_DIR=/opt/steamvr/drivers/frame_controller/resources/rendermodels
cmake --build build
```

The default source directory is `/opt/steamvr/drivers/frame_controller/resources/rendermodels`. It must contain the `frame_controller_left` and `frame_controller_right` directories. The build runs the converter, which checks the generated GLB files and profile node references, then places `profile.json` and the two GLBs under `share/wivrn/assets/controllers/steam-frame`. The install step copies that asset tree to WiVRn's normal data directory.

The option is off by default, so ordinary builds do not need SteamVR or Steam Frame assets. Enable it only for a Linux client build intended for Steam Frame: it defaults that client to the `steam-frame` visual profile. Leave it disabled for generic Linux builds and other headsets.

The generated GLBs embed geometry and texture data read from the local SteamVR installation. The converter contains no model or texture data. Do not commit or redistribute generated GLBs unless you have confirmed that you have permission to do so.

The profile uses WiVRn's existing OpenXR visual-response handling. Standard trigger, squeeze, thumbstick, and A/B/X/Y responses continue to use the controller inputs supported by SteamVR's Oculus Touch compatibility mode. The additional Frame D-pad, bumper, view, menu, and system parts animate when the runtime exposes their Frame controller input paths; those responses may remain still while the controllers use an emulated interaction profile that does not expose them. This option changes the headset's visual profile selection and does not change the streaming protocol.
