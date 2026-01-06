import importlib.metadata

try:
	__version__ = importlib.metadata.version("Ryan-GPT-systems")
except importlib.metadata.PackageNotFoundError:
	try:
		__version__ = importlib.metadata.version("ryan-gpt-systems")
	except importlib.metadata.PackageNotFoundError:
		__version__ = "0.0.0"

# Distributed training exports
from ryan_gpt_systems.ddp_bucket import DDPBucketed
from ryan_gpt_systems.ddp_flat import DDPIndividualParameters
from ryan_gpt_systems.optimizer_state_sharding import ShardedOptimizer
from ryan_gpt_systems.distributed_training import (
    setup_distributed,
    cleanup_distributed,
    get_strategy,
    DistributedTrainer,
    DistributedDataLoader,
    is_main_process,
    get_rank,
    get_world_size,
)

# Tensor parallelism exports
from ryan_gpt_systems.tensor_parallelism import (
    TensorParallelGroup,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    TensorParallelAttention,
    TensorParallelMLP,
    TensorParallelTransformerBlock,
    TensorParallelTransformerLM,
    copy_to_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)