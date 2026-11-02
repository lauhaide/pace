CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m flashrag.retriever.index_builder \
    --retrieval_method bge \
    --model_path BAAI/bge-large-en-v1.5 \
    --corpus_path /mnt/data/obqa/corpus/crag/CRAG_all_corpus_250_trec_cast.jsonl \
    --save_dir /mnt/data/obqa/corpus/crag \
    --max_length 400 \
    --batch_size 2048 \
    --pooling_method cls \
    --faiss_type Flat \
    --save_embedding

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m flashrag.retriever.index_builder \
#    --retrieval_method bge \
#    --model_path BAAI/bge-large-en-v1.5 \
#    --corpus_path /mnt/data/obqa/corpus/browsecomp/browsecomp-plus-passage-tevatron.jsonl \
#    --save_dir /mnt/data/obqa/corpus/browsecomp \
#    --max_length 400 \
#    --batch_size 2048 \
#    --pooling_method cls \
#    --faiss_type Flat \
#    --save_embedding #\

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m flashrag.retriever.index_builder \
#    --retrieval_method bge \
#    --model_path BAAI/bge-large-en-v1.5 \
#    --corpus_path /nfs/public/datasets/kilt/chunk_100_bge_en_v1.5/kilt_knowledgesource_r1_search100.jsonl \
#    --save_dir /mnt/data/obqa/corpus/kilt/chunk_100_bge_en_v1.5 \
#    --max_length 300 \
#    --batch_size 2048 \
#    --pooling_method cls \
#    --faiss_type Flat \
#    --save_embedding #\     