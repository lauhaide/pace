# 这里我假定了所有的index都是ivf量化的
#GPU

CORPUS="kilt" # Other corpora are "browsecomp" and "crag"

# The following parameters are to hold everyting corpus/index and retriever models in the same GPU e.g., H200 w/ 140Gb
CUDA_VISIBLE_DEVICES=0 python -u codebase/retrieval/wiki_corpus_load_rerank.py $CORPUS 5003 0 0 monot5l

# The following parameters are to hold corpus/index in GPUs 0 and 1; and retriever models in GPU 2 (e.g., H100 w/ 80Gb). In this cofig. coundlt make the model to share the GPUs :(
#CUDA_VISIBLE_DEVICES=0,1,2 python -u codebase/retrieval/wiki_corpus_load_rerank.py $CORPUS 5003 0,1 2  monot5l & 

