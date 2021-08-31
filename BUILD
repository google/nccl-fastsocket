load("@rules_pkg//:pkg.bzl", "pkg_deb", "pkg_tar")

# Dual-licensed, using the least restrictive per go/thirdpartylicenses#same
licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "nccl_utilities",
    srcs = [
        "utilities.cc",
    ],
    hdrs = [
        "utilities.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@nccl//:plugin_lib",
    ],
)

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
        ":nccl_utilities",
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

genrule(
    name = "gen_triggers",
    outs = ["triggers"],
    cmd = "echo 'activate-noawait ldconfig' > $@",
)

pkg_deb(
    name = "package-deb",
    architecture = "amd64",
    data = ":tarball",
    description = "Fast Socket for NCCL 2",
    maintainer = "Chang Lan <changlan@google.com>",
    package = "google-fast-socket",
    recommends = ["libnccl2"],
    triggers = ":gen_triggers",
    version = "0.0.4",
)

pkg_tar(
    name = "tarball",
    extension = "tar.gz",
    deps = [
        ":doc",
        ":lib",
    ],
)

pkg_tar(
    name = "lib",
    srcs = [":libnccl-net.so"],
    mode = "0644",
    package_dir = "/usr/lib/",
)

pkg_tar(
    name = "doc",
    srcs = [":LICENSE"],
    mode = "0644",
    package_dir = "/usr/share/doc/google-fast-socket/",
)
