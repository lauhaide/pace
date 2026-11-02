import argparse  # 解析命令行参数
from datetime import datetime  # 处理日期时间
import os
from typing import List  # 类型注解

import ray  # 导入 Ray 进行分布式训练
import torch  # PyTorch 深度学习框架
from ray.util.placement_group import placement_group  # Ray 的 placement group 用于 GPU 资源分配

# 导入 PPO 训练相关的 Ray Actor
from openrlhf.trainer.ray import (
    ActorModelRayActor,  # 处理 Actor 模型的 Ray Actor
    CriticModelRayActor,  # 处理 Critic 模型的 Ray Actor
    PPORayActorGroup,  # PPO 训练中使用的 Ray Actor 组
    ReferenceModelRayActor,  # 处理参考模型（Reference Model）的 Ray Actor
    RewardModelRayActor,  # 处理 Reward Model（奖励模型）的 Ray Actor
    create_vllm_engines,  # 创建 vLLM 引擎的函数
)
from openrlhf.utils import get_strategy  # Function to obtain training policy

# Define a reward calculation function (example) that takes reward values ​​calculated from multiple Reward Models and sums them.
# This function can be replaced with a user-defined reward function.
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)  # Multiple reward tensors are summed, with the summation occurring along the first dimension, as the first dimension represents different reward calculation methods.

# Functions for parsing and validating input parameters
def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node  # Calculate the total number of Actor processes

    # Ensure that rollout_batch_size is divisible by actor_world_size.
    assert (
        args.rollout_batch_size % actor_world_size == 0
    ), f"The rollout_batch_size must be divisible by actor_world_size. (Current value): {args.rollout_batch_size}, {actor_world_size}"

    # Ensure that ZeRO-3 can only be used when vLLM is enabled.
    assert args.zero_stage != 3 or args.vllm_num_engines > 0, "ZeRO-3 only supports use when vLLM is enabled."

    # Ensure that the number of Actors is an integer multiple of the number of vLLM engines.
    if args.vllm_num_engines > 0:
        assert (
            actor_world_size % args.vllm_num_engines == 0
        ), f"actor_world_size must be an integer multiple of vllm_num_engines, current value: {actor_world_size}, {args.vllm_num_engines}"

    # If Critic pre-training is enabled, ensure that the number of Actor processes is an integer multiple of the number of Critic processes.
    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, current value: {actor_world_size}, {critic_world_size}"

# training main function
def train(args):
    _validate_args(args)  # Verify parameters first

    strategy = get_strategy(args)  # Get training strategy

    # If the Actor and Reference Model need to share GPU resources, create a placement group.
    pg = None
    # if args.colocate_actor_ref or args.colocate_all_models:
    #     assert (
    #         args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node #
    #     )  #"当 Actor 和 Reference Model 共享 GPU 时，它们的 num_nodes 和 num_gpus_per_node 必须相同"   共享 GPU Reference 模型和 Actor 模型始终在同一设备，避免 GPU 间通信开销（提高吞吐）
    #     bundles = [
    #         {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
    #         for _ in range(args.actor_num_nodes)
    #     ]
    #     pg = placement_group(bundles, strategy="STRICT_SPREAD")  # 创建 GPU 资源组
    #     print("gpu资源组",bundles)
    #     ray.get(pg.ready())  # 等待 GPU 资源分配完成 #pg.ready() 会 阻塞代码运行，直到 Ray 成功分配所有资源。
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # Initialize the vLLM engine (for text generation)
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len  # Calculate the maximum sequence length
        if args.colocate_all_models:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,  # vLLM Number of engines
            args.vllm_tensor_parallel_size,  # vLLM Tensor Parallel size
            args.pretrain,  # Pre-trained model path
            args.seed,  # random seed
            args.enable_prefix_caching,  # Enable prefix caching?
            args.enforce_eager,  # Should Eager execution mode be enabled?
            max_len,  # The maximum length generated
            args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
            pg if args.colocate_all_models else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
        )
    print('vllm_engines initialization complete')
    print(os.system('nvidia-smi'))
    # The Ray Actor group creates the Actor model.
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=(0.5 if args.colocate_all_models else 0.80) if pg else 1,  # If a shared GPU is used, each actor uses 0.65 GPUs; otherwise, the default is 1 GPU.
    )
    # Create the Ray Actor group for the Reference Model
    ref_model = PPORayActorGroup(
        args.ref_num_nodes, # How many nodes are there in total?
        args.ref_num_gpus_per_node, #How many GPUs per node
        ReferenceModelRayActor, # which model to use
        pg=pg, #Here, pg is passed to PPORayActorGroup to ensure that the Actor runs according to the scheduled resources.  
        num_gpus_per_actor=0.20 if pg else 1,  # If a shared GPU is used, each reference model will only use 0.25 GPU units.
    )
    print('actor_model and ref_model initialization complete')

    pg = None
    # if args.critic_pretrain and args.colocate_critic_reward:
    #     assert (
    #         args.critic_num_nodes == args.reward_num_nodes
    #         and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
    #     ), f"当 Critic 和 Reward Model 共享 GPU 时，它们的 num_nodes 和 num_gpus_per_node 必须相同"

    #     bundles = [
    #         {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node}
    #         for _ in range(args.critic_num_nodes)
    #     ]
    #     pg = placement_group(bundles, strategy="STRICT_SPREAD")
    #     ray.get(pg.ready())
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # If Critic pre-training is enabled, then create the Ray Actor group for the Critic Model.
    if args.critic_pretrain:
        critic_model = PPORayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )
    else:
        critic_model = None

    # Multiple Reward Models
    if not args.remote_rm_url:
        reward_pretrains = args.reward_pretrain.split(",")
        reward_models = [
            PPORayActorGroup(
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )
            for _ in reward_pretrains
        ]
    else:
        reward_models = None
    # Initialize the reference model, reward model, and actor model.
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))  # The advantage of asynchronous loading for initializing the reference model is that all models load simultaneously without waiting for each other; otherwise, with synchronous loading, one model would have to wait for the next.
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))  # Initialize the Actor model
    if not args.remote_rm_url:
        for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
            refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))  # Initialize reward model

    results = ray.get(refs)  # Waiting for model initialization to complete # In the Ray distributed computing framework, ray.get() is used to block the main process, wait for the remote task (Ray Actor or task) to complete execution, and obtain the return result.
    all_success = all(result for result in results if result is not None)
    print(f"all_success: {all_success}")
    os.system("nvidia-smi")
    # If using Critic pre-training, initialize the Critic model.
    if args.critic_pretrain:
        # Critic training depends on the maximum number of steps (max_step), therefore Critic initialization must be performed after Actor initialization.
        # TODO: Consider using the first Reward Model as the Critic Model.
        max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())  # Get the maximum number of steps in Actor training.
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))  # Initialize the Critic model
        ray.get(refs)  # Waiting for the Critic model to complete initialization.

    # Training Actor and Critic models
    refs = actor_model.async_fit_actor_model(
        critic_model,  # Critic model
        ref_model,  # Reference Model
        reward_models,  # reward model
        args.remote_rm_url,  # Remote Reward Model API (if available)
        reward_fn=reward_fn,  # reward calculation function
        vllm_engines=vllm_engines,  # vLLM engine (for text generation)
        remote_sufficient_url=args.remote_sufficient_url,
    )
    ray.get(refs)  # Wait for training to complete

    # Save Actor Model
    ray.get(actor_model.async_save_model())

    # If you are using Critic pre-trained and need to save the value network, then save the Critic model.
    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())



# def train(args):
#     _validate_args(args)

#     # configure strategy
#     strategy = get_strategy(args)

#     # if colocated, create placement group for actor and ref model explicitly.
#     pg = None
#     if args.colocate_actor_ref:
#         assert (
#             args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
#         ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

#         bundles = [
#             {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
#             for _ in range(args.actor_num_nodes)
#         ]
#         pg = placement_group(bundles, strategy="STRICT_SPREAD")
#         ray.get(pg.ready())



#     actor_model = PPORayActorGroup(
#         args.actor_num_nodes,
#         args.actor_num_gpus_per_node,
#         ActorModelRayActor,
#         pg=pg,
#         num_gpus_per_actor=0.75 if pg else 1,
#     )

#     ref_model = PPORayActorGroup(
#         args.ref_num_nodes,
#         args.ref_num_gpus_per_node,
#         ReferenceModelRayActor,
#         pg=pg,
#         num_gpus_per_actor=0.25 if pg else 1,
#     )

#     # if colocated, create placement group for critic and reward model explicitly.
#     pg = None
#     if args.critic_pretrain and args.colocate_critic_reward:
#         assert (
#             args.critic_num_nodes == args.reward_num_nodes
#             and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
#         ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

#         bundles = [
#             {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node}
#             for _ in range(args.critic_num_nodes)
#         ]
#         pg = placement_group(bundles, strategy="STRICT_SPREAD")
#         ray.get(pg.ready())

#     if args.critic_pretrain:
#         critic_model = PPORayActorGroup(
#             args.critic_num_nodes,
#             args.critic_num_gpus_per_node,
#             CriticModelRayActor,
#             pg=pg,
#             num_gpus_per_actor=0.75 if pg else 1,
#         )
#     else:
#         critic_model = None

#     # multiple reward models
#     if not args.remote_rm_url:
#         reward_pretrains = args.reward_pretrain.split(",")
#         reward_models = []
#         for _ in reward_pretrains:
#             reward_models.append(
#                 PPORayActorGroup(
#                     args.reward_num_nodes,
#                     args.reward_num_gpus_per_node,
#                     RewardModelRayActor,
#                     pg=pg,
#                     num_gpus_per_actor=0.25 if pg else 1,
#                 )
#             )
#     else:
#         reward_models = None

#     # 初始化参考模型（Reference Model）、奖励模型（Reward Model）和 Actor 模型
#     refs = []
#     refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))  # 初始化参考模型
#     refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))  # 初始化 Actor 模型
#     if not args.remote_rm_url:
#         for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
#             refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))  # 初始化奖励模型

#     # 初始化 vLLM 引擎（用于文本生成）
#     vllm_engines = None
#     if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
#         max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len  # 计算最大序列长度
#         vllm_engines = create_vllm_engines(
#             args.vllm_num_engines,  # vLLM 引擎数量
#             args.vllm_tensor_parallel_size,  # vLLM Tensor 并行大小
#             args.pretrain,  # 预训练模型路径
#             args.seed,  # 随机种子
#             args.enable_prefix_caching,  # 是否启用前缀缓存
#             args.enforce_eager,  # 是否强制启用 Eager 执行模式
#             max_len,  # 生成的最大长度
#         )

#     ray.get(refs)  # 等待模型初始化完成

#     # 如果使用 Critic 预训练，则初始化 Critic 模型
#     if args.critic_pretrain:
#         # Critic 训练依赖于最大步数（max_step），因此必须在 Actor 初始化完成后再进行 Critic 初始化
#         # TODO: 考虑使用第一个 Reward Model 作为 Critic Model
#         max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())  # 获取 Actor 训练的最大步数
#         refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))  # 初始化 Critic 模型
#         ray.get(refs)  # 等待 Critic 模型初始化完成

#     # 训练 Actor 和 Critic 模型
#     refs = actor_model.async_fit_actor_model(
#         critic_model,  # Critic 模型
#         ref_model,  # Reference Model
#         reward_models,  # 奖励模型
#         args.remote_rm_url,  # 远程 Reward Model API（如果有）
#         reward_fn=reward_fn,  # 奖励计算函数
#         vllm_engines=vllm_engines,  # vLLM 引擎（用于文本生成）
#     )
#     ray.get(refs)  # 等待训练完成

#     # 保存 Actor 模型
#     ray.get(actor_model.async_save_model())

#     # 如果使用 Critic 预训练且需要保存价值网络，则保存 Critic 模型
#     if args.critic_pretrain and args.save_value_network:
#         ray.get(critic_model.async_save_model())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--scheduler_type", type=str, default="cosine_with_min_lr", help="scheduler type")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--use_kl_estimator_k2", action="store_true", default=False, help="Use the k2 estimator in http://joschu.net/blog/kl-approx.html")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce","reinforce_baseline", "rloo", "group_norm", "group_norm_pos", "group_norm_token_efficiency"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--remote_sufficient_url", type=str, default=None, help="remote sufficient API (HTTP)")

    #
    parser.add_argument("--naive_grpo", action="store_true", default=False, help="Whether to use only outcome (verifiable) reward")
    parser.add_argument("--reward_shape", action="store_true", default=False, help="Whether to do reward shaping")
    parser.add_argument("--reshape_method", type=str, default="intermediate")
    parser.add_argument("--reshape_filter", action="store_true", default=False, help="Whether to do reward shaping")
    parser.add_argument("--diversity_level", type=str, default=None, help="compute diversity of sampled rollouts (rollout/step/None--default)")
    parser.add_argument("--adaptive", action="store_true", default=False, help="whether to train with shallow/deep search modes.")
    parser.add_argument("--deep_penalty", type=float, default=-1, help="penalty when using deep search and not needed.")
    parser.add_argument("--approx_coverage", type=str, default=None, help="a file with silver passages")
    parser.add_argument("--with_process_r", action="store_true", default=False, help="whether to train with trajectory information reward (average of potentials).")
    parser.add_argument("--with_group_d", action="store_true", default=False, help="whether to train with trajectory diversity reward.")   
    parser.add_argument("--goal_state_mod", action="store_true", default=False, help="special modulation for final state, goal or not goal.")
    parser.add_argument('-ir', '--intrinsic', action="append", default=[], help="intrinsic rewards to use options: [rets, acov, sdiv, gola, nite, tips].")
    parser.add_argument("--omega_u", type=float, default=0.2, help="utility weight.")
    parser.add_argument("--omega_c", type=float, default=0.1, help="coverage weight.")
    parser.add_argument("--omega_d", type=float, default=0.05, help="state diversity weight.")
    parser.add_argument("--lambda_expl", type=float, default=0.1, help="group diversity reward weight.")
    parser.add_argument("--process_r_lambda_phi", type=float, default=0.8, help="Average Phis reward weight.")
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha weight in potential term.")

    


    
    
    parser.add_argument("--prompt_type", type=str, default="default")
    #parser.add_argument("--prompt_template_type", type=str,  required=True)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")
    
    # Filter parameters
    parser.add_argument("--enable_accuracy_filter", action="store_true", default=False)
    parser.add_argument("--freezing_filter_steps", type=int, default=-1)
    parser.add_argument("--accuracy_lower_bound", type=float, default=-1, help="Lower bound for accuracy")
    parser.add_argument("--accuracy_upper_bound", type=float, default=0.9, help="Upper bound for accuracy")


    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)
    # sht update
    parser.add_argument("--apply_uncompleted_filter", action="store_true", default=False)
    parser.add_argument("--apply_query_filter", action="store_true", default=False)
    parser.add_argument("--apply_select_response_by_prm", action="store_true", default=False)
    parser.add_argument("--apply_select_response_longer_pos", action="store_true", default=False)
    parser.add_argument("--group_method", type=str, choices=['group_reward_incomplete_equal_to_neg', 'group_reward_with_learn_mask', 'normal'])
    parser.add_argument("--use_length_reward_in_efficiency", action="store_true", default=False)
    # End of UPDATE
    # ZBC UPDATE
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )

    parser.add_argument("--random_temperature", action="store_true", default=False)
    # End of UPDATE
    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        args.enable_prefix_caching = False
        print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache.")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."   
    if os.environ.get('RAY_LOCAL_MODE', '1') == '0':
        ray.init(local_mode=True) #Recall local debug status
    train(args)
