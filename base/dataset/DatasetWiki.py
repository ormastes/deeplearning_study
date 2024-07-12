from datasets import load_dataset
# /workspace/data
# /workspace/dev
def load_wikipedia_dataset(cache_dir="/workspace/data/cache/wikipedia"):
    return load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir, trust_remote_code=True)

if __name__ == "__main__":
    load_wikipedia_dataset()