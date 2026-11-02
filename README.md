# PACE-GRPO

This repository contains resources for our paper [Action-Level Credit Assignment for Retrieval-Augmented Reasoning](https://aclanthology.org/2026.findings-emnlp.1203/). Please contact me at lperez@ed.ac.uk for any question.


## Environment and Libraries used

Our experiments are run with this docker image:
```
docker pull lauhaide/algomo:v1
```

In addition we install libraries with:
```
pip install -r codebase/requirements.txt
apt-get -y install cuda-toolkit-12-4 --fix-missing
```

## Retrieval Corpus and Data

__Download and chunking into passages__ For the Kilt Wikipedia-based corpus we follow instructions in the [TIRESRAG‑R1](https://github.com/probe2/TIRESRAG-R1) repository, both for chunking.
For BrowseCompPlus, we used the passage pre-processed corpus from [(Meng et a., 2026)](https://arxiv.org/abs/2602.21456). For CRAG, we download the corpus from [here (Task #3)](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md) and pre-processed it by extending the [TREC CAsT Tools](https://github.com/grill-lab/trec-cast-tools)'s functionality (see instructions in [this folder](codebase/evaluation/FlashRAG/scripts/trec-cast-tools.md)).

__Retrieval Index Building__ We follow instructions and scripts from [TIRESRAG‑R1](https://github.com/probe2/TIRESRAG-R1). You can find our scripts [here](codebase/evaluation/FlashRAG/scripts/).
```
bash build_index.sh
```

Under [the data folder](codebase/data/) you will find the evaluation sets as well as the training data. The scripts for the off-line annotations (i.e., silver passages and state 0 potential's scores) can be found [here](codebase/retrieval/).

## Retrieval Serivice

To launch the retrieval server, index and retrievers you need to run the following script:
```
scripts/wiki_service.sh
```
Note: this script contains examples on how to also launch the service with BrowseCompPlus and CRAG corpora.

## Reward API Services

You need to use the following script to launch the reward assessment API:
```
scripts/answer_reflection_reward.sh
```
Note that if you train [TIRESRAG‑R1](https://github.com/probe2/TIRESRAG-R1) the you first need to launch ```scripts/sufficient_thinking_reward.sh```.

## Training

The example script to launch the training process for PACE-GRPO can be found in:
```
scripts/qwen_grpo.sh
```

Most of the training settings are shared by all baselines and ablations. The flags that define each implemented approach are as described in [these instructions](codebase/scripts/training_variants.md).


## Evaluation

An example script on how to run the evaluation process can be found in:
```
scripts/qwen_grpo.sh
```
Note: Inside there are further comments on options for parameters.  
Note: Configuration parameters for the evalution can be found in [this .yaml file](codebase/evaluation/src/).


## 📖 Citation

If you use PACE-GRPO or some of our existing methods' re-implementations in your research, please cite:

```bibtex
@inproceedings{perez-beltrachini-lapata-2026-action,
    title = "Action-Level Credit Assignment for Retrieval-Augmented Reasoning",
    author = "Perez-Beltrachini, Laura  and Lapata, Mirella",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026"
}
```

## 🙏 Acknowledgments

We would like to thank the following open-source projects and their authors for providing the foundation of this work:

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) for the retrieval and evaluation framework
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for the reinforcement learning framework
- [TIRESRAG‑R1](https://github.com/probe2/TIRESRAG-R1) for the base code
- [TREC CAsT Tools](https://github.com/grill-lab/trec-cast-tools) to chunk BrowseCompPlus and CRAG