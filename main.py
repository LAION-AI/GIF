import webdataset as wds
from torch.utils.data import DataLoader
from config import dataset_urls, batch_size, cache_path, log_path, max_concurrent_processes, job_name, comment, n_nodes, gpus, cpus_per_gpu, ntasks_per_node, output_path
import subprocess
import os
import time
from tqdm import tqdm

def _start_job(sbatch_file):
    """start job"""
    args = ["sbatch"]
    args.append(sbatch_file)
    sbatch_output = subprocess.check_output(args).decode("utf8")
    lines = sbatch_output.split("\n")

    lines = [line for line in lines if "Submitted" in line]
    if len(lines) == 0:
        raise ValueError(f"slurm sbatch failed: {sbatch_output}")

    parsed_sbatch = lines[0].split(" ")
    job_id = parsed_sbatch[3].strip()
    return job_id

def _run_job(sbatch_file):
    """
    Run a job and wait for it to finish.
    """
    try:
        job_id = _start_job(sbatch_file)
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        return "exception occurred"
    else:
        return job_id

def _generate_sbatch(log_path, current_shard_number, job_name):
        """
        Generate sbatch for a worker.
        sbatch: allows you to specify a configuration and task in a file
            - https://slurm.schedmd.com/sbatch.html
        """

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --comment={comment}
#SBATCH --nodes={n_nodes}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --output={log_path}/slurm-%x_%j.out
python3 slurm_job.py {current_shard_number}
"""

def _create_sbatch_and_run(cache_path, log_path, job_name, current_shard_number):
    """
    Create a sbatch file, submit it to slurm, and wait for it to finish.
    """

    # make the filenames unique using the current timestamp
    sbatch_script_path = os.path.join(cache_path, f"sbatch_script_{current_shard_number:06d}.sh")

    # save the file to the cache path
    with open(sbatch_script_path, "w", encoding="utf-8") as sbatch_file:
        sbatch_file.write(
            _generate_sbatch(log_path, current_shard_number, job_name)
        )

    # now we need to run the job
    _run_job(sbatch_script_path)

def main(
    dataset_urls, 
    batch_size,
    cache_path,
    log_path,
    output_path,
    max_concurrent_processes,
    job_name,
    ntasks_per_node
    ):
        ### Creating log and cache directory
        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        for i in tqdm(range(20)):
            job_pending = True
            while job_pending:
                current_number_of_jobs = len(subprocess.check_output(["squeue", "-n", job_name]).decode("utf8").split('\n')) - 2
                if (current_number_of_jobs < max_concurrent_processes):
                    _create_sbatch_and_run(cache_path, log_path, job_name, i)
                    job_pending = False
                else:
                    time.sleep(0.5)

if __name__ == "__main__":
    main(dataset_urls, batch_size, cache_path, log_path, output_path, max_concurrent_processes, job_name, ntasks_per_node)