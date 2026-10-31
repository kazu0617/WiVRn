# WiVRn tracing server baseline build

- Result: `wivrn-server` build succeeded (cmake exit 0; final action linked `server/wivrn-server`). No client, server library, dashboard, CLI, or test targets were requested.
- Source commit: `fc48b6155ebb4fd658ddb8f8b2006086659747e4` in detached scratch worktree `work/wivrn-tracing`.
- Applied PKGBUILD patch stack, in order:
  1. `steam-frame-controller.patch` — SHA-256 `e2a7b96bfa96f0a18d8f6abc6ab55592584e7f6ae32b4ea473c4f3b7a7c0cb9b`
  2. `wivrn-separate-eye-swapchains-8531-builddefault.patch` — SHA-256 `76666eddc1ca64ec09321951b694a20650af5f710ddfebf39651e828b9f11da9`
  3. `wivrn-controller-offset-streamgui-fc48-diagnostic.patch` — SHA-256 `ee47eb1cd6d63c588b0dddc09a22bbd0c23833c8c320ae217549678b5f23d5ec`
- Perfetto SDK: official v58.2 `perfetto-cpp-sdk-src.zip`, extracted under `work/perfetto-sdk-v58.2/sdk`; archive SHA-256 `e56126ff81f0f914d35fe03c72419d147acf2890f36fbda7d4558fd5b84c4b09`. The upstream SDK docs describe this two-file amalgamated source SDK as the supported integration route: https://github.com/google/perfetto/blob/main/docs/instrumentation/tracing-sdk.md
- Configure options included `WIVRN_USE_PERFETTO=ON`, `PERFETTO_SDK_DIR=work/perfetto-sdk-v58.2/sdk`, `WIVRN_TRACE_MONADO=OFF`, server on, and client/server-library/dashboard/wivrnctl/tests off. Build command: `nice -n 19 ionice -c3 cmake --build work/wivrn-tracing/build-server --target wivrn-server --parallel 2`.
- Binary: `work/wivrn-server-tracing-baseline`; SHA-256 `31c153f2c6124a5272a3ba90500b91043b013795f84e3c023fc8d32655132c61` (x86-64 ELF PIE). The original build-tree executable has the same hash.
- Logs: configure `work/wivrn-tracing-configure.log`; full build `work/wivrn-tracing-build.log`.

For a controlled capture, `WIVRN_TRACING=inprocess` enables the in-process backend without a Perfetto daemon; set `WIVRN_TRACING_FILE` to a writable output path (supports `%t` timestamp and `%p` PID; unset writes `wivrn-<timestamp>-<pid>.pftrace` in the working directory). The trace file is written when the session flushes or shuts down. `WIVRN_TRACING_BUFFER_KB` is optional and defaults to 65536. `WIVRN_TRACING=system` or `system+inprocess` additionally requires a reachable Perfetto `traced` producer endpoint at `$XDG_RUNTIME_DIR/wivrn-perfetto-producer.sock` and a consumer session. No launch, service, or device operation was performed.
