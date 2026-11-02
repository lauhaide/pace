from tqdm import tqdm
import json
import jsonlines
import requests

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_file_json(data, fp):
    with open(fp, 'w') as f:
        json.dump(data, f)       


batch_size = 200
url_wiki = "http://0.0.0.0:5003/queries"  # Retrieve server address

input_data = load_file("codebase/data/training_set/train-orlhf-musique.jsonl")
print(len(input_data))
result_fp = "codebase/data/train-orlhf-musique-ga.jsonl"
topk=20

new_data = []
for idx in tqdm(range(len(input_data) // batch_size)):
    batch = input_data[idx*batch_size:(idx+1)*batch_size]
    #query_list=[item['question'] + item['answers'][0] for item in batch]
    query_list=[item['question'] + item['answer'] for item in batch]
    strategy_list = ['Deep'] * len(query_list)
    response = requests.post(url_wiki, json={"queries": query_list, "k": , "rerank": 1, "rescore": 1, "strategy": None}) 
    print(response.status_code)
    if response.status_code == 200:
        print('Query complete, query successful.')
        result = response.json()
        answers = result["answers"]
        distances = result["distances"]
        for ps, ds, it in zip(answers, distances, batch):
            context = []
            for e in range(topk):
                context.append({'idx':e, 'title': ps[e].split('\n')[0], 'text': ps[e].split('\n')[1], 'score': ds[e]})
            it['ctxs'] = context
            new_data.append(it)

if len(input_data) % batch_size > 0:
    batch = input_data[(idx+1)*batch_size:]
    #query_list=[item['question'] + item['answers'][0] for item in batch]
    query_list=[item['question'] + item['answer'] for item in batch]
    response = requests.post(url_wiki, json={"queries": query_list, "k": topk, "rerank": 1, "rescore": 1, "strategy": None}) 
    if response.status_code == 200:
        print('Query complete, query successful.')
        result = response.json()
        answers = result["answers"]
        distances = result["distances"]
        for ps, ds, it in zip(answers, distances, batch):
            context = []
            for e in range(topk):
                context.append({'idx':e, 'title': ps[e].split('\n')[0], 'text': ps[e].split('\n')[1], 'score': ds[e]})
            it['ctxs'] = context
            new_data.append(it)


save_file_jsonl(new_data, result_fp)
print('Files saved to, ', result_fp)

