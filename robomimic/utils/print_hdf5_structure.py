import h5py


# path = "datasets/__test__/test_data.hdf5"
# path = "datasets/__test__/test_data_realworld_fix-action-dim_20260817.hdf5"
# path = "datasets/source/coffee.hdf5"

path = "datasets/square/ph/image_v15.hdf5"
# path = "datasets/square/ph/low_dim_v15.hdf5"
# path = "datasets/transport/ph/image_v15.hdf5"
# path = "datasets/square/ph/low_dim_v15.hdf5"
# path = "datasets/square/ph/demo_v15.hdf5"


def print_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    else:
        print(f"{indent}[Dataset] {name} shape={obj.shape}")

        # # remove
        # if len(obj.shape) <=2: 
        #     import numpy as np
        #     np.set_printoptions(precision=4, suppress=True)
        #     for i in range(3):
        #         print(f"\t\t\t {obj[i]}")


with h5py.File(path, "r") as f:
    print("[ROOT]")
    
    # 최상위 구조
    for k in f.keys():
        print(f"  {k}/")

    # 대표 demo 하나만 보기
    demo = sorted(f["data"].keys())[0]
    print(f"Total num demonstrations: {len(f['data'].keys())}")
    print("\n")
    print(f"\n[Structure of data/{demo}]")

    f[f"data/{demo}"].visititems(print_structure)
