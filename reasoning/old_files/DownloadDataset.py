import datasets
ds = datasets.load_dataset("Nutanix/CPP-UNITTEST-BENCH")
ds.save_to_disk("/workspace/data/model/reasoning/raw/cpp_ut_bench")
ds = datasets.load_dataset("Elfsong/Mercury")
ds.save_to_disk("/workspace/data/model/reasoning/raw/CompCodeVet")