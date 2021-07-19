# Dual-licensed, using the least restrictive per go/thirdpartylicenses#same
licenses(["notice"])

exports_files(["LICENSE"])

# Faster socket plugin for NCCL applications.
cc_library(
    name = "plugin",
    srcs = [
        "compat.cc",
        "net_fastsocket.cc",
    ],
    hdrs = [
        "compat.h",
    ],
    # Export the symbol containing the NCCL plugin vtable so it can be
    # loaded at runtime via dlopen + dlsym. This means we also need to
    # always link this library, otherwise it'll be dropped at build time.
    linkopts = [
        "-Wl,--export-dynamic-symbol=ncclNetPlugin_v4",
        "-Wl,--export-dynamic-symbol=ncclNetPlugin_v3",
        "-Wl,--export-dynamic-symbol=ncclNetPlugin_v2",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@nccl//:plugin_lib",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libnccl-net.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        ":plugin",
    ],
)
