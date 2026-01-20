from huggingface_hub import snapshot_download

repo_id = "minjae-chung/alphadta"
local_dir = "../data"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False, 
)
print("Download Complete:", local_dir)


# tar --zstd -xf /scratch/dcgm-mj/AlphaDTA/data/af3_embedding/pdbcleansplit_only/emb-000000.tar.zst \
#   -C /scratch/dcgm-mj/AlphaDTA/data/af3_embedding/pdbcleansplit_only
