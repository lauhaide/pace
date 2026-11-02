import itertools
import math
import os
import socket
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
from transformers.trainer import get_scheduler
from datetime import timedelta
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import init_process_group, torch_dist_barrier_and_cuda_sync

from .launcher import BasePPORole
from .utils import get_physical_gpu_id


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None, #远程 Reward Model API。
        critic_train_remote: bool = False, #是否远程训练 Critic Model。
        remote_sufficient_url: str = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs) # # 调用 PPOTrainer 的初始化
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote
        self.remote_sufficient_url = remote_sufficient_url
        self.experience_maker = RemoteExperienceMaker(  #负责 PPO 经验收集（使用 Actor 交互环境，计算 Reward）。
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.remote_sufficient_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0: #仅 主进程（rank 0） 负责管理 vLLM 权重同步。
            master_address = ray._private.services.get_node_ip_address() #获取主机 IP 和可用端口，用于 进程间通信。
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            debug_mode = os.environ.get("DEBUG_MODE", "1") == "0"

            if debug_mode:
                print("🛠️ Running in debug mode: using world_size = 0")
                world_size = 1
            else:
                world_size = vllm_num_engines * vllm_tensor_parallel_size + 1 #计算 进程总数：1 个主进程 + vllm_num_engines * vllm_tensor_parallel_size 个 vLLM 进程。
                print(f"🚀 Running in full mode: world_size = {world_size}")
            backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
            print('打印vllm engine信息',self.vllm_engines)
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)

            if not debug_mode:
                refs = [
                    engine.init_process_group.remote(
                        master_address,
                        master_port,
                        0 if debug_mode else i * vllm_tensor_parallel_size + 1,
                        world_size,
                        "openrlhf",
                        backend=backend,
                        use_ray=use_ray
                    )
                    for i, engine in enumerate(self.vllm_engines)
                ]
            # ray.get(refs)  # 添加这行！
            # 远程初始化 vLLM 进程组（engine.init_process_group.remote(...)）。
            timeout = timedelta(minutes=2)
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name= "openrlhf")
                self._model_update_group =  "openrlhf"
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name="openrlhf",
                    timeout=timeout
                )
            #init_process_group(...) 创建 vLLM 分布式通信组。
            if not debug_mode:
                ray.get(refs)
            # Wait for all vLLM process groups to complete initialization.
        torch.distributed.barrier()
        # Synchronize all processes to ensure that all processes complete vLLM process group initialization.

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()

        # 2. triger remote critic model training
        if self.critic_train_remote:
            critic_status_ref = self.critic.fit.remote()
        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            status = super().ppo_train(global_steps)

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                if (os.environ.get("DEBUG_MODE", "1") == "0"):
                    return  status
                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._broadcast_to_vllm()
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        else:
            status = {}

        # 5. wait remote critic model training done
        if self.critic_train_remote:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _broadcast_to_vllm(self):
        if os.environ.get("DEBUG_MODE", "1") == "0":
            print("Debug mode: directly updating vLLM engine parameters")
            
            # Obtain the state dictionary of the actual model
            state_dict = self.actor.model.module.state_dict()
            
            # Create update task list
            update_refs = []
            for engine in self.vllm_engines:
                ref = engine.update_model_directly.remote(state_dict)
                update_refs.append(ref)
            
            # Waiting for all updates to complete
            results = ray.get(update_refs)
            
            # Check if the update was successful.
            if not all(results):
                print("Warning: Some vLLM engines failed to update")
            return 
        # Clear the GPU cache to prevent OutOfMemoryError (OOM).
        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        
        # Traverse all parameters of the model
        for name, param in model.named_parameters():
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            count += 1  # A counter is used to determine whether it is the last parameter.
            if not self.use_cuda_ipc:

                # Broadcasts are only executed in the main process (rank 0).
                if torch.distributed.get_rank() == 0:
                    # To obtain the parameter shape, for ZeRO-3, use ds_shape.
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    # Notify all VLLM engines to prepare to receive updates.
                    refs = [
                        engine.update_weight.remote(name, dtype=param.dtype, shape=shape, 
                            empty_cache=count == num_params)  # The last parameter clears the cache.
                        for engine in self.vllm_engines
                    ]
                # The debug section will likely also throw an error.
                    # ray.get(refs)
            # ZeRO-3 Special handling: The parameters for each shard need to be collected first.
                # if torch.distributed.get_rank()== 0:
                #     print("model_update_group size:", torch.distributed.get_world_size(group=self._model_update_group))
                torch.distributed.barrier()
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank()== 0:
                        # Broadcast parameters to all VLLM engines
                        if use_ray:
                            import ray.util.collective as collective
                            # print(f" 广播权重shape {param.data.shape}, device {param.data.device}, dtype {param.data.dtype}")
                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            # print('打印param data 参数',param.data.shape, param.data.device, param.data.dtype)
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        # Waiting for all engine updates to complete 
                                # Ensure the broadcast is complete
                    torch.distributed.barrier()
                    
                    # Verify successful update
                    if torch.distributed.get_rank() == 0:
                        ray.get(refs)
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)
                        # print(f"要更新的参数名字 {name}: mean={weight.float().mean().item()}, std={weight.float().std().item()}, shape={param.shape}")
                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch_dist_barrier_and_cuda_sync()

                    # ray.get(refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if self.critic_train_remote:
            ref = self.critic.save_checkpoint.remote(tag)
        # self.strategy.save_ckpt( #I don’t want to save the gradient information for the moment. It’s too troublesome to resume training or something.
        #     self.actor.model,
        #     os.path.join(args.ckpt_path, "_actor"),
        #     tag,
        #     args.max_ckpt_num,
        #     args.max_ckpt_mem,
        #     client_states,
        # )
        self.strategy.save_model(self.actor.model, self.tokenizer, os.path.join(args.ckpt_path, f"_actor_model_{tag}"))
        # wait
        if self.critic_train_remote:
            ray.get(ref)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            print('Set environment variables')
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
        # Configure the NCCL environment variable to disable the CUMEM function to prevent NCCL synchronization from hanging.
        
        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
        )
        #Load pre-trained Actor models, supporting LoRA/FlashAttention/DeepSpeed.
        strategy.print(actor)

        # configure tokenizer 初始化 Tokenizer。
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None
        # 如果启用了 EMA（指数滑动平均），则创建一个额外的模型副本。
        # configure optimizer 初始化优化器。    
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)#向上取整
        self._max_steps = max_steps
        if args.scheduler_type == "cosine_with_min_lr":
            actor_scheduler = get_scheduler(
                "cosine_with_min_lr",
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
            )
        elif args.scheduler_type == "constant":
            actor_scheduler = get_scheduler(
                "constant",
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
            )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model: #如果启用了 EMA，则保存 EMA 版本的模型（self.ema_model）。否则，直接保存当前 actor。这样，最终用于推理的模型是 EMA 版本，而不是原始训练模型。
            # ema模型就是计算之前步的模型参数+ 当前模型参数 ，起一个平滑作用。
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        # kill
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True
        )
        print("self.prompts_dataset[0:2]:", self.prompts_dataset[0:2])
        print("self.prompts_dataloader:", self.prompts_dataloader)
        # kill

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
        remote_sufficient_url: str = None,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            remote_sufficient_url=remote_sufficient_url,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()


        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )
