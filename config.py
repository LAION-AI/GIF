## Logpath
cache_path = './cache'
log_path = './logs'
output_path = './output'

## slurm configuration
job_name = 'inferencepipeline'
comment = 'laion'
n_nodes = 1
gpus = 1
cpus_per_gpu = 2
ntasks_per_node = 8
max_concurrent_processes = 8

## Webdataset being processed
batch_size = 2
dataset_urls = []
for i in range(231350):
    dataset_urls.append(f's3://s-datasets/laion5b/laion2B-data/{i:06d}.tar')
dataset_urls = [f'pipe:aws s3 cp {url} -' for url in dataset_urls]