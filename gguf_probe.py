import struct

path = "models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf"

with open(path, "rb") as f:
    magic = f.read(4)
    assert magic == b"GGUF", magic
    version, = struct.unpack("<I", f.read(4))
    tensor_count, = struct.unpack("<Q", f.read(8))
    kv_count, = struct.unpack("<Q", f.read(8))
    print(f"version={version} tensor_count={tensor_count} kv_count={kv_count}")

    def read_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")

    SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

    def skip_value(vtype):
        if vtype == 8:
            read_str()
        elif vtype == 9:
            atype, = struct.unpack("<I", f.read(4))
            alen, = struct.unpack("<Q", f.read(8))
            for _ in range(alen):
                skip_value(atype)
        else:
            f.read(SIZES[vtype])

    for _ in range(kv_count):
        key = read_str()
        vtype, = struct.unpack("<I", f.read(4))
        skip_value(vtype)

    for i in range(tensor_count):
        name = read_str()
        n_dims, = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{n_dims}Q", f.read(8*n_dims))
        dtype_tag, = struct.unpack("<I", f.read(4))
        offset, = struct.unpack("<Q", f.read(8))
        if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
            print(name, "dims=", dims, "dtype_tag=", dtype_tag)
