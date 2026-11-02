To add scripts for CRAG corpus:

1. Clone [trec-cast-tools](https://github.com/grill-lab/trec-cast-tools)
2. Copy our added scripts:
```
cp codebase/evaluation/FlashRAG/scripts/CRAG_generator.py YOUR-CLONE-DIR/trec-cast-tools/corpus_processing/generators
cp codebase/evaluation/FlashRAG/scripts/main_trec.py YOUR-CLONE-DIR/trec-cast-tools/corpus_processing
```
3. Add the following line to YOUR-CLONE-DIR/trec-cast-tools/corpus_processing/generators/\__init\__.py
```
from .CRAG_generator import CRAG_Generator
```
4. Run 
```
cd YOUR-CLONE-DIR/trec-cast-tools/corpus_processing
python3 main_trec.py 
   --process_crag 
   --output_type jsonlines 
   --output_dir YOUR-DATA-DIR/corpus/crag 
   --duplicates_file YOUR-DATA-DIR/corpus/crag/all_duplicates.txt 
   --crag_collection YOUR-DATA-DIR/corpus/crag/crag_task_3_dev_v4/crag_task_3_dev_v4_0.jsonl
```