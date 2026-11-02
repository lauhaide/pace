import argparse
from hashlib import md5
from pathlib import Path

from generators import CRAG_Generator
from passage_chunkers import PassageChunker
from utils import process_batch

parser = argparse.ArgumentParser(
    description='Collection Processing Parameters')

# CRAG collection path    
parser.add_argument(
    '--crag_collection',
    type=str,
    # if collection dowloaded with bash script
    default="/mnt/data/obqa/corpus/crag/crag_task_1_and_2_dev_v4.jsonl",
    help="Path to the raw CRAG collection"
)
# Duplicates file path
parser.add_argument(
    '--duplicates_file',
    type=str,
    required=True,
    help="Path to duplicates file"
)
parser.add_argument('--batch_size', type=int, default=100000,
                    help="Number of documents per batch")
parser.add_argument('--process_crag', default=False, action='store_true')
parser.add_argument('--output_dir', type=str, required=True,
                    help="Directory to write files to")
parser.add_argument('--output_type', type=str, default="jsonlines",
                    help="Output file type: trecweb or jsonlines")


if __name__ == '__main__':

    args = parser.parse_args()
    passage_chunker = PassageChunker()
    output_path = f"{args.output_dir}/{args.output_type}"
    md5_dir_path = f"{args.output_dir}/md5_hashes"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(md5_dir_path).mkdir(parents=True, exist_ok=True)

    if args.process_crag:
        print("--- Processing CRAG ---")

        crag_generator: CRAG_Generator = CRAG_Generator(
            args.crag_collection, args.duplicates_file, args.batch_size
        ).generate_documents()

        process_batch(
            collection_name='CRAG_Paral9',
            generator=crag_generator,
            passage_chunker=passage_chunker,
            output_type=args.output_type,
            output_path=output_path,
            md5_dir_path=md5_dir_path
        )
