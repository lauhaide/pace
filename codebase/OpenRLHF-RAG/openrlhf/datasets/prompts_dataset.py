from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, prompt_type=None, model_name=None) -> str:
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]
        answer = data["answer"]
        if not prompt_type or prompt_type=="ours":
            messages_chat=[
                {"role": "system", "content":  "You are a helpful assistant. Answer the given question. \
                You must reason **clearly and completely** inside <think> and </think> before providing any final answer. \
                Always identify and verify all key entities (e.g., person names, locations, dates, awards) mentioned in the question. \
                If you are uncertain about an entity or fact, or if the question requires external knowledge, you may use \
                <search>your query</search>, and the top search results will be returned between <information> and </information>. Carefully read and reflect on each newly retrieved piece of information. \
                You can search as many times as you want. \
                When reasoning, you must ensure your reasoning path aligns strictly with the evidence. \
                After reasoning, before providing your final answer, rethink it to make sure the answer is exactly correct for the original question. \
                Use the most accurate span from the evidence when possible. \
                Only after satisfying all the above, give the final answer inside <answer> and </answer>. For example, <answer> Beijing </answer>. \
                After outputting the final answer in <answer>  </answer>, You have one chance to reflect on the answer. If you choose not to reflect, nothing further needs to be done.\
                Otherwise, you can then re-examine your thinking process, the information obtained, and even search for more information to verify your previous answer or correct the previous answer. Remember, after reflection ends, you should output the  answer in <answer>  </answer>."
                }, 
                {"role": "user", "content":question}
            ]      
        elif prompt_type=="trajel3":
                messages_chat=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the given question. \
You must reason **clearly and completely** about all pieces of information required to arrive at the final answer. \
Enclose your reasoning inside <think> and </think> tags. \
If you are uncertain about an entity or fact, or if the question requires external knowledge, you may \
need to issue simple sub-questions for external search to gather information. \
In the end, the sequence of simple sub-questions, gathered information, and reasoning steps should lead to the final answer. \n\n \
Simple sub-questions can be of the following types: \n \
- **parallel** (gather independent pieces of information for sub-parts of the question), \n \
- **sequential** (re-write a preceding question based on a previously gathered piece of information), \n \
- **interpretation** (re-write a preceding question for each possible concrete enty to which ambiguous entity mentions in the history may refer to), or \n \
- **clarification** (gather information about relevant entities mentioned in the history to make sure reasoning is linked to appropriate concrete entities). \n \
If a sub-question fails to gather the necessary information, reformulate it in one of the following ways:\n \
- **paraphrase** (using different words), \n \
- **general** (making it more general or vague --drop some details), \n \
- **specific** (making it more specific --add relevant details from the history), or \n \
- **reflect** (making it completely different --reflect about exploring different information gathering paths). \n \
Re-think the entire information search starategy if you are unable to locate the necessary information. \
Consider different ways the question could be understood. \
Look for potential ambiguities in terminology, scope, or intent. \n\n \
Choose one of the above sub-question types; \
then generate the corresponding sub-question enclosed within <search>your sub-question</search> tags. \
**Carefully think about the search strategy** for the sub-question. \
The search strategy can be one of the following:\n \
- Shallow: information can be found with a cheap similarity search over the external resource \n \
- Deep: information needs a deep search against the external resource. \n \
Generate the strategy enclosed within <strategy> and </strategy> tags.\n \
Next to each sub-question, a set of passages will be provided enclosed within <information> and </information> tags.\n\n \
Please **do not repeat simple questions** you have already issued, as this is a waste of time. \
Please **do not use correference** such as 'this' within simple questions, always use the entity mention as introduced before. \
Carefully reflect on the formulated simple questions and each piece of information within the set of provided passages. \
After you think you have gathered all the neccessary information, before providing your final answer, rethink it to make \
sure the answer is exactly correct for the original question. \
Only after satisfying all the above, give your final short phrase answer inside <answer> and </answer> tags.\n\n \
Output format:\n\n \
<answer> your short phrase answer </answer>\n\n \
or \n\n \
<strategy> only one of: Shallow or Deep </strategy> <search>your sub-question</search>. \n\n"
            }, 
                    {"role": "user", "content":question}
                ]
        else:
            print('Error undefined prompt!')
            
        prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"

    else:

        base_prompt = """Answer the given question. \
            You must reason **clearly and completely** inside <think> and </think> before providing any final answer. \
            Always identify and verify all key entities (e.g., person names, locations, dates, awards) mentioned in the question. \
            If you are uncertain about an entity or fact, or if the question requires external knowledge, you may use \
            <search>your query</search>, and the top search results will be returned between <information> and </information>. Carefully read and reflect on each newly retrieved piece of information. \
            You can search as many times as you want. \
            When reasoning, you must ensure your reasoning path aligns strictly with the evidence. \
            After reasoning, before providing your final answer, rethink it to make sure the answer is exactly correct for the original question. \
            Use the most accurate span from the evidence when possible. \
            Only after satisfying all the above, give the final answer inside <answer> </answer>. For example, <answer> Beijing </answer>. \
            After outputting the final answer in <answer> </answer>, You have one chance to reflect on the answer. If you choose not to reflect, nothing further needs to be done.\
            Otherwise, you can then re-examine your thinking process, the information obtained, and even search for more information to verify your previous answer or correct the previous answer. Remember, after reflection ends, you should output the  answer in <answer>  </answer>.
            
            Question:{question}
            
            \n<think>"""
        question = data["question"]
        idx = data["idx"]
        answer = data["answer"]
        prompt = base_prompt.format(question=question)

    return str(idx) + "<|idx_prompt_split|>" + prompt + "<pro_answer>" + answer


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        prompt_type = getattr(self.strategy.args, "prompt_type", None)
        print(f"Using instruction: {prompt_type}")
        model_name = getattr(self.strategy.args, "pretrain", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, prompt_type, model_name)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
