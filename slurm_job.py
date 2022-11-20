import fire
import webdataset as wds
from torch.utils.data import DataLoader
from config import cache_path, output_path
import os

def worker(current_shard):

    # Read environment variables
    # These are set by slurm_distributor or SLURM itself
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    print(f'global_rank: {global_rank}, local_rank: {local_rank}')
    try:
        dataset_url = f'pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{current_shard:06d}.tar -'

        ds = wds.WebDataset(dataset_url, handler=wds.ignore_and_continue)
        dl = DataLoader(ds, num_workers=1, batch_size=1)

        c = 0

        for _ in dl:
            c += 1

        print(f"{current_shard:06d}: {c}")

        os.makedirs(output_path, exist_ok=True)
        
        with open(f'{output_path}/{current_shard:06d}.txt', "w") as f:
            f.write(f"{current_shard:06d}: {c}")


    except Exception as e:
        os.rename(cache_path + f"/sbatch_script_{current_shard:06d}.sh", cache_path + f"/sbatch_script_{current_shard:06d}_failed.sh")
        try:
            with open("error_logs.txt", "a") as f:
                f.write(f"Shard number {current_shard:06d}:" + e)
        except:
            print('Could not write error logs...')
    else:
        os.remove(cache_path + f"/sbatch_script_{current_shard:06d}.sh")
    

if __name__ == "__main__":
    fire.Fire(worker)

