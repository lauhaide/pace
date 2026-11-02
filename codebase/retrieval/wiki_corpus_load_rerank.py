
from socket import ntohl
from flask import Flask, request, jsonify
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import os
import time
import sys
import argparse
import json 
from tqdm import tqdm
import os
import sys
import time
import psutil
import threading
from datetime import datetime
import torch 
import time
import gc 
import threading

import pynvml

from reranker.monoT5 import MonoT5
import pandas as pd

from reranker.vallina_bert import VanillaBert
from reranker.bert_ranker import BertRanker

search_lock = threading.Lock()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # 转换为GB

def format_size(size_bytes):
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def loading_monitor():
    """监控加载进度和资源使用"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    while loading:
        current_memory = get_memory_usage()
        elapsed_time = time.time() - start_time
        memory_change = current_memory - start_memory
        
        sys.stdout.write('\r' + ' ' * 80 + '\r')  # 清除当前行
        sys.stdout.write(
            f"Loading... Time elapsed: {elapsed_time:.1f}seconds | "
            f"memory usage: {current_memory:.1f}GB (Increase: {memory_change:+.1f}GB) | "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        sys.stdout.flush()
        time.sleep(1)
def read_index_with_monitor(index_path, gpus_for_index=None):
    """Index loading function with monitoring"""
    global loading
    loading = True
    # 检查是否存在预训练的IVF索引
    ivf_index_path = index_path 
    if "ivf"  in index_path.lower(): #os.path.exists(ivf_index_path):
        print(f"\nDiscover pre-trained IVF indexes and load them directly: {ivf_index_path}")
        try:
            cpu_ivf_index = faiss.read_index(ivf_index_path)
            print("预训练索引加载成功！")
            nlist = int(np.sqrt(cpu_ivf_index.ntotal))  # 确保nlist在所有路径中都定义
        except Exception as e:
            print(f"预训练索引加载失败: {e}，将重新训练索引")
            cpu_ivf_index = None
    elif "flat" in index_path.lower():
        cpu_ivf_index = faiss.read_index(index_path)
        nlist = int(np.sqrt(cpu_ivf_index.ntotal))  # 确保nlist在所有路径中都定义

    else:
        cpu_ivf_index = None

    # 获取文件信息
    file_size = os.path.getsize(index_path)
    print(f"\nStart loading index file:")
    print(f"file path: {index_path}")
    print(f"file size: {format_size(file_size)}")
    print(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"initial memory usage: {get_memory_usage():.1f}GB")
    print("\nLoading progress:")

    # 启动监控线程
    monitor_thread = threading.Thread(target=loading_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    print('INDEX CLASS', type(cpu_ivf_index).__name__)
    print('Trained?', cpu_ivf_index.is_trained)


    try:
        # 加载索引
        start_time = time.time()
        print("1. Load CPU index...")
        
        # 读取基础索引
        if cpu_ivf_index is None:
            cpu_index = faiss.read_index(index_path)
            dimension = cpu_index.d
            ntotal = cpu_index.ntotal
            # 配置IVF参数
            nlist = int(np.sqrt(ntotal))  # 聚类中心数量，根据数据量自动调整
            print(f"\n2. 配置IVF索引 (nlist={nlist})...")

            res = faiss.StandardGpuResources()
            gpu_quantizer = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))

            # 创建新的IVF索引
            # quantizer = faiss.IndexFlatL2(dimension)
            gpu_ivf_index = faiss.IndexIVFFlat(gpu_quantizer, dimension, nlist)
            # 训练索引
            print("\n 3. 训练IVF索引...")
            train_size = min(500000, ntotal)  # 最多使用50万个向量训练

            # 收集训练数据
            train_vectors = []
            train_indices = np.random.choice(ntotal, train_size, replace=False)
            # 确保 train_indices[0] 不大于 ntotal - train_size
            while train_indices[0] > ntotal - train_size - 1:
                train_indices = np.random.choice(ntotal, train_size, replace=False)

            if hasattr(cpu_index, 'reconstruct_n'):
                print('Use reconstruct_batch to batch obtain vector 1')
                train_vectors = cpu_index.reconstruct_n(
                    int(train_indices[0]), 
                    len(train_indices)
                )
            else:
                train_vectors = np.vstack([
                    cpu_index.reconstruct(int(idx)).reshape(1, -1)
                    for idx in train_indices
                ])
            # for idx in train_indices:
            #     if hasattr(cpu_index, 'reconstruct'):
            #         vec = cpu_index.reconstruct(int(idx))
            #         train_vectors.append(vec)
            
            # train_vectors = np.array(train_vectors)
            gpu_ivf_index.train(train_vectors)
            del train_vectors  # 释放训练数据内存
            gc.collect()
            # 将索引转回CPU进行添加操作

            print("5. Return the index to the CPU for vector addition....")
            cpu_ivf_index = faiss.index_gpu_to_cpu(gpu_ivf_index)
            del gpu_ivf_index  # 释放GPU索引内存
            del res
            gc.collect()
            print("4. Add vectors to IVF index in batches...")
            batch_size = 2000000
            for start_idx in tqdm(range(0, ntotal, batch_size)):
                end_idx = min(start_idx + batch_size, ntotal)
                
                # 使用reconstruct_batch批量获取向量
                if hasattr(cpu_index, 'reconstruct_n'):
                    print('使用reconstruct_batch批量获取向量')
                    batch_vectors = cpu_index.reconstruct_n(start_idx, end_idx - start_idx)
                else:
                    # 降级方案：如果没有reconstruct_batch方法
                    batch_vectors = np.vstack([
                        cpu_index.reconstruct(int(i)).reshape(1, -1)
                        for i in range(start_idx, end_idx)
                    ])
                
                # 添加批次到索引
                cpu_ivf_index.add(batch_vectors)
                del batch_vectors  # 释放批次向量内存     
                gc.collect()   
            # 释放原始索引内存

            # 保存CPU版本的索引
            print("\n7. 保存训练好的IVF索引...")
            faiss.write_index(cpu_ivf_index, ivf_index_path)
            print(f"索引已保存到: {ivf_index_path}")

            print(f"当前内存使用: {get_memory_usage():.1f}GB")

            del cpu_index
            gc.collect()
            print(f"当前内存使用: {get_memory_usage():.1f}GB")

        print("\n5. Prepare GPU resources...")
        #Create GPU resources
        if not gpus_for_index:
            gpus_for_index = range(torch.cuda.device_count())

        gpu_resources = []
        for i in gpus_for_index:  # Using GPU 0 and 1
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)

        print("\n6. Configure GPU options...")
        co = faiss.GpuMultipleClonerOptions()
        #co.useFloat16 = True ### not sure adding this, but index_builder gpu-train uses as this
        co.shard = True  # GPU partitioning
        co.usePrecomputed = True  # No pre-calculation, saving GPU resources

        print("\n7. Transfer IVF indexes to multi-GPU...")
        print('faiss.get_num_gpus()', faiss.get_num_gpus())
        gpu_list = [i for i in gpus_for_index]
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(
            gpu_resources,  # GPU resource list
            index = cpu_ivf_index,     # CPU index
            co = co            # GPU options
        )
        #            gpus = gpu_list,        # GPU IDs
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_ivf_index, co) #Transfer directly to available GPU resources
        del cpu_ivf_index
        gc.collect()
        # Set search parameters
        if("ivf" in index_path):
            nprobe = min(500, nlist//10)  # By default, 10% of cluster centers are checked, but no more than 30.
            gpu_index.nprobe = nprobe
            print(f"\n8. Set search parameters (nprobe={nprobe})")

        loading = False
        monitor_thread.join()
        
        # Calculate the final state
        end_time = time.time()
        total_time = end_time - start_time
        final_memory = get_memory_usage()
        
        # Print completion information
        print("\n\nLoading completed:")
        print(f"total time: {total_time:.1f}second ({total_time/60:.1f}minute)")
        print(f"final memory usage: {final_memory:.1f}GB")
        print(f"Index configuration:")
        print(f"- total number of vectors: {gpu_index.ntotal}")
        print(f"- vector dimensions: {gpu_index.d}")
        print(f"- number of cluster centers(nlist): {nlist}")
        # print(f"- Search cluster number(nprobe): {nprobe}")
        print(f"completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print GPU usage
        for i in gpu_list:
            print(f"\nGPU {i} Video memory usage: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB (allocated) {torch.cuda.memory_reserved(i)/1024**3:.2f}GB (reserved)")
        os.system("nvidia-smi")
        
        # Release CPU index memory
        # Empty the cache
        torch.cuda.empty_cache()

        return gpu_index

    except Exception as e:
        loading = False
        monitor_thread.join()
        print(f"\n\nLoading failed: {str(e)}")
        raise


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = []
        total = sum(1 for _ in file)
        file.seek(0)    
        for line in tqdm(file, total=total, desc="Loading corpus"):
            data = json.loads(line)
            corpus.append(data['contents'])
    return corpus

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    # 从请求中获取查询向量
    data = request.json
    
    embed_start = time.time()
    print("\n1. Generate query vector...")

    queries = data["queries"]
    strategy = data["strategy"]
    pen_th = data["pen_th"] if 'pen_th' in data.keys() else 0.05
    pen_th_high = data["pen_th_high"] if 'pen_th_high' in data.keys() else 0.4
    rerank = data.get("rerank", 0)
    rescore = data.get("rescore", 0)
    return_k = data.get("k", 3)
    k = return_k
    k_deep = data.get("k_deep", 50)
    if rerank == 1:
        k = 100
    
    # s = time.time()
    print('Now embed', queries)
    for name, param in model.model.named_parameters():
        print(f"parameter {name} in device: {param.device}")
        break  # Print only the first parameter as an example
    query_embeddings = model.encode_queries(queries)
    #query_embeddings = model.encode(queries,  batch_size=len(queries), is_query=True) #TODO batch?
    print('queries embedded...')

    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()

    query_embeddings = query_embeddings.astype('float32')
    print(f"Vector generation complete, time taken: {time.time() - embed_start:.2f}seconds.")
    print(f"vector shape: {query_embeddings.shape}")

    if strategy:
        # split query embeddings into shallow and deep search
        shallow_ids = [e for e, stra in enumerate(strategy) if stra == 'Shallow' ]
        deep_ids = [e for e, stra in enumerate(strategy) if stra == 'Deep' ]
        deep_query_embeddings = query_embeddings[deep_ids]
        query_embeddings = query_embeddings[shallow_ids]
    else:
        shallow_ids, deep_ids = [e for e, _ in enumerate(queries)], []

    deep_penalty = []
    all_answers = []
    all_distances = []
    search_start = time.time()
    print(f"\n2. Shallow search for similar documents...[{query_embeddings.shape[0]}]")

    # Add batch processing logic
    batch_size = 16  # This value can be adjusted based on GPU memory usage.     
    total_queries = query_embeddings.shape[0]
    all_D = []
    all_I = []

    for i in range(0, total_queries, batch_size):
        end_idx = min(i + batch_size, total_queries)
        batch_queries = query_embeddings[i:end_idx]
        #print('batch_queries', batch_queries)
        
        # 添加错误处理
        try:
            with search_lock:
                batch_D, batch_I = index.search(batch_queries, k=k)
           # batch_D, batch_I = index.search(batch_queries, k=k)
            all_D.append(batch_D)
            all_I.append(batch_I)
        except Exception as e:
            print(f"批处理搜索错误 (索引 {i}~{end_idx}): {str(e)}")
            # 降级到CPU搜索作为备选方案
            try:
                print("尝试使用CPU进行搜索...")
                cpu_index = faiss.index_gpu_to_cpu(index)
                batch_D, batch_I = cpu_index.search(batch_queries, k=k)
                all_D.append(batch_D)
                all_I.append(batch_I)
            except Exception as e2:
                print(f"CPU搜索也失败: {str(e2)}")
                # 填充-1表示搜索失败
                batch_D = np.full((end_idx - i, k), -1.0, dtype=np.float32)
                batch_I = np.full((end_idx - i, k), -1, dtype=np.int64)
                all_D.append(batch_D)
                all_I.append(batch_I)

    # Merge batch results
    if all_D and all_I:
        D = np.vstack(all_D) if len(all_D) > 1 else all_D[0]
        I = np.vstack(all_I) if len(all_I) > 1 else all_I[0]
    elif total_queries > 0:
        # If all batches fail, create a result of all -1s.
        D = np.full((total_queries, k), -1.0, dtype=np.float32)
        I = np.full((total_queries, k), -1, dtype=np.int64)
    else:
        I = None # all requests where for strategy = Deep

    batch_size_deep = 8
    total_queries_deep = 0
    if strategy and len(deep_ids) > 0:
        print(f"\n3. Deep search for similar documents...[{deep_query_embeddings.shape[0]}]")
        total_queries_deep = deep_query_embeddings.shape[0]
        all_D_deep = []
        all_I_deep = []

        for i in range(0, total_queries_deep, batch_size_deep):
            end_idx = min(i + batch_size_deep, total_queries_deep)
            batch_queries = deep_query_embeddings[i:end_idx]
            #print('batch_queries', batch_queries)
            
            # 添加错误处理
            try:
                with search_lock:
                    batch_D, batch_I = index.search(batch_queries, k_deep)
            # batch_D, batch_I = index.search(batch_queries, k=k)
                all_D_deep.append(batch_D)
                all_I_deep.append(batch_I)
            except Exception as e:
                print(f"批处理搜索错误 (索引 {i}~{end_idx}): {str(e)}")
                # 降级到CPU搜索作为备选方案
                try:
                    print("尝试使用CPU进行搜索...")
                    cpu_index = faiss.index_gpu_to_cpu(index)
                    batch_D, batch_I = cpu_index.search(batch_queries, k_deep)
                    all_D_deep.append(batch_D)
                    all_I_deep.append(batch_I)
                except Exception as e2:
                    print(f"CPU搜索也失败: {str(e2)}")
                    # 填充-1表示搜索失败
                    batch_D = np.full((end_idx - i, k), -1.0, dtype=np.float32)
                    batch_I = np.full((end_idx - i, k), -1, dtype=np.int64)
                    all_D_deep.append(batch_D)
                    all_I_deep.append(batch_I)

        # Merge batch results
        if all_D_deep and all_I_deep:
            D_deep = np.vstack(all_D_deep) if len(all_D_deep) > 1 else all_D_deep[0]
            I_deep = np.vstack(all_I_deep) if len(all_I_deep) > 1 else all_I_deep[0]
        else:
            # If all batches fail, create a result of all -1s.
            D_deep = np.full((total_queries_deep, k), -1.0, dtype=np.float32)
            I_deep = np.full((total_queries_deep, k), -1, dtype=np.int64)

    print(f"Search complete, time taken: {time.time() - search_start:.2f} seconds.")

    # Add result check
    valid_results = ((I != -1).sum()  if total_queries>0 else 0) + ((I_deep != -1).sum() if total_queries_deep>0 else 0)
    total_expected =  (total_queries * k) + (total_queries_deep * 100)
    print(f"Valid results: {valid_results}/{total_expected} ({valid_results/total_expected*100:.1f}%)")

    cnt, cnt_deep = 0, 0
    for q_idx in range(total_queries + total_queries_deep):
    #for q_idx, (idx, distance) in enumerate(zip(I, D)):
        if q_idx in shallow_ids:
            idx = I[cnt]
            distance = D[cnt]
            deep_search = False
            cnt +=1
            print(f"Shallow Query {q_idx+1}: (out of {k}) retrieved idx: {idx[:min(k,10)]} retrieved dist: {distance[:min(k,10)]}")
            current_k = k
        elif q_idx in deep_ids:
            idx = I_deep[cnt_deep]
            distance = D_deep[cnt_deep]            
            deep_search = True
            cnt_deep +=1
            print(f"Deep Query {q_idx+1}: (out of {k_deep}) retrieved idx: {idx[:min(k_deep,10)]} retrieved dist: {distance[:min(k_deep,10)]}")
            current_k = k_deep

        # Add index validation to ensure the system does not crash due to invalid indexes.
        answers = []
        distances = []
        for i, v in zip(idx[:current_k], distance[:current_k]):
            if i >= 0 and i < len(corpus):
                answers.append(corpus[i])
                distances.append(str(v))
            else:
                answers.append("未找到匹配文档")
                distances.append(0)

        if rerank == 1 or rescore == 1 or deep_search:
            if reranker_model == 'monot5l':
                lines = []
                for e, a in enumerate(answers):
                    lines.append({
                        'query_id': e,
                        'docid': e,
                        'query': queries[q_idx],
                        'text': a,
                    })

                df = pd.DataFrame(lines)
                df = reranker.transform(df)

                #print(f"Writing reranked results to {args.rerank_output_path}")
                new_distances = []
                for _, row in df.iterrows():
                    new_distances.append((row['prob'], row['text']))
                
            elif reranker_model == 'utility':
                whole_answers = [a for a in answers]
                question = [queries[q_idx]]*len(whole_answers)
                new_distances = reranker.get_utilities(test_data=whole_answers, question=question)
                print('utilities', new_distances)
                new_distances = [(sigmoid(d), a) for d, a in zip(new_distances, whole_answers)]
           
            if rerank == 1 or deep_search:
                original_topk = new_distances[:return_k]
                new_distances.sort(key=lambda x: x[0], reverse=True)
                new_distances = new_distances[:return_k]
                # does deep search improve retrieval results?
                avg_ori = sum([d for d , _ in original_topk])/return_k
                avg_new = sum([d for d , _ in new_distances])/return_k
                print(f'[th:{pen_th}/{pen_th_high}] new', avg_new, 'ori', avg_ori)
                score_diff = avg_new - avg_ori
                pen = int(deep_search and (score_diff < pen_th or score_diff > pen_th_high))
                deep_penalty.append(pen)
                answers = [a for _ , a in new_distances] # take answers again as have been sorted
            else:
                deep_penalty.append(0)

            distances = [d for d , _ in new_distances]
        else:
            deep_penalty.append(0) #no penalties in this setting

        all_distances.append(distances)
        all_answers.append(answers)

        if total_queries_deep > 100:
            gc.collect()
        
    assert total_queries + total_queries_deep == cnt + cnt_deep

    return jsonify({"queries": queries, "answers": all_answers, "distances": all_distances, "deep_penalty": deep_penalty})


if __name__ == "__main__":
    data_type = sys.argv[1]
    port = sys.argv[2]
    gpus_for_index = [int(x) for x in sys.argv[3].split(',')]
    gpu_retriever = sys.argv[4]
    reranker_model = sys.argv[5]
    print('Start building')
    print('Available GPUs: ', torch.cuda.device_count())

    model = FlagModel(
        "BAAI/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
        devices=[f'cuda:{gpu_retriever}']
        #device="cuda:1", # manually indicate the GPU devices: devices=['cuda:0', 'cuda:1']
    )

    print("The model has been loaded.")
    print(f'GPU index in: {gpus_for_index}, Retriever in: {gpu_retriever}')

    if reranker_model == 'monot5l':
        reranker = MonoT5(model="castorini/monot5-large-msmarco", tok_model="t5-large", batch_size=32)
        print('MonoT5-L loaded')
    elif reranker_model == 'utility':
        model_dir = '/mnt/data/algomo/codebase/data'

        pretrained_model = 'bert-base-uncased'
        base_model = BertRanker(pretrained_model)
        base_model.to('cuda')

        # this is the Passage Utility re-ranking model from https://openreview.net/forum?id=JLkgI0h7wy
        reranker = VanillaBert(base_model, device='cuda', pretrained_model=pretrained_model)
        reranker.load_state_dict(torch.load(os.path.join(
            model_dir, 'qwen3-4b-it-unified-vanilla_bert_3epoch_32-0.001-TITLEbis-1acc_LM-0.25be-combinedbest-val-acc-model.pt'))['state_dict'])
        print('Utility Ranker Loaded')

    # 加载语料库
    if data_type =="kilt":
        file_path ="/nfs/public/datasets/kilt/chunk_100_bge_en_v1.5/kilt_knowledgesource_r1_search100.jsonl"
    elif data_type =="browsecomp":
        file_path ="/mnt/data/obqa/corpus/browsecomp/browsecomp-plus-passage-tevatron.jsonl"
    elif data_type =="crag":
        file_path ="/mnt/data/obqa/corpus/crag/CRAG_all_corpus_250_trec_cast.jsonl"

    corpus = load_corpus(file_path)

    print(f"The corpus has been loaded.-{len(corpus)}")

    # 加载建好的索引
    if data_type =="kilt":
        index_path ="/mnt/data/obqa/corpus/kilt/chunk_100_bge_en_v1.5/bge_Flat.index"
    else:
        index_path =f"/mnt/data/obqa/corpus/{data_type}/bge_Flat.index" # for the other datasets
    
    try:
        print("\nPrepare to load index...")
        index = read_index_with_monitor(index_path, gpus_for_index=gpus_for_index)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    print("The index has been created.")

    #for name, param in model.model.named_parameters():
    for name, param in model.model.named_parameters():
        print(f"parameter {name} in device: {param.device}")
        break  # Print only the first parameter as an example

    app.run(host="0.0.0.0", port=port, debug=False)  # 在本地监听端口5003
    print("You can start querying")

