import h5py

path = "datasets/square/ph/image_v15_pause_IS_Cphase.hdf5"


def print_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    else:
        print(f"{indent}[Dataset] {name} shape={obj.shape}")

with h5py.File(path, "r") as f:
    print("[ROOT]")
    
    # 최상위 구조
    for k in f.keys():
        print(f"  {k}/")

    # 대표 demo 하나만 보기
    demo = sorted(f["data"].keys())[0]
    print(f"\n[Structure of data/{demo}]")

    f[f"data/{demo}"].visititems(print_structure)