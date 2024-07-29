import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from utils import load_file, PROMPT_DICT, save_file_jsonl, preprocess_input_data, postprocess_answers_closed, TASK_INST
from metrics import metric_max_over_ground_truths, exact_match_score, match
from hclu import get_enquery
import ast
import backoff
import openai
import zhipuai
import json
from openai.error import APIError, Timeout, APIConnectionError
import scipy.stats as stats
from modelscope import AutoTokenizer, AutoModel, snapshot_download

openai.api_base = " "
openai.api_key = " "

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def completions_instructgpt_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def get_index_number(l,n,mode):
    result=[]
    if mode == 'average':
        ave=l/n
        i=0
        while i<l:
            result.append[i]
            i+=ave
    elif mode == 'guass':
        lower, upper = 0, l
        mu, sigma = 0,1
        # X表示含有最大最小值约束的正态分布
        # N表示不含最大最小值约束的正态分布
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
        a = X.rvs(n)  # 取其中的b个数，赋值给a；a为array类型
        result=[int(i) for i in a]
        while 1:
            x=result[0]
            result=result[1:]
            if x in result:
                x+=1
                result.append(x)
            else:
                result.append(x)
            set_result=set(result)
            if len(result)==len(set_result):
                break
    elif mode=='top':
        result=range(5)
    print(result)
    return result

def llm_evi_ans(model,evidences,query):
    result=[]
    #result.append(evidences[0])
    for evi in evidences:
        try:
            rel=get_llm_chatglm(evi,query).upper()
        #rel = get_offline_llm(model,evi, query)[0].upper()
            print(rel)
            if 'False' in rel:
                continue
            else:
                result.append(evi)
        except:
            result.append(evi)
    return result

def get_offline_llm(model,evi,query):
    ins1="Do the following sentences have the similar meaning in specific actions? Please think carefully, we originally tended to believe that they were defferent. Only say 'true' when you believe firmly that they are the same."
    ins2="Just say 'true' if they do; otherwise just say 'false'. The response is either 'true' or 'false'. Only output one word. No other words. The response can only be 'true' or 'false'."
    prompt = '###Instruction:\n'+ins1+ins2+ '\n\n###Question:\n' + evi + '\n' + query +'\n\n### Response:\n'
    preds,_=call_model(prompt,model,50)
    #print(preds)
    return preds


def get_llm_gpt(evi,query):
    #prompt='请判断以下论述是否相似，如果相似则输出True，如果不相似则输出False'
    prompt = ('Are the following statements relevant or not? Just say true if they are; otherwise just say false. The output is either true or false. Only output one word.')

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": evi+'\n'+query}
        ],
        temperature=0.2,
        max_tokens=20,
    )
    return (completion.choices[0].message['content'])

def get_llm_chatglm(evi,query): #思维链表达VMD
    # your api key
    #print("llm!")
    zhipuai.api_key = " "
    ins_0='You are a cautious language assistant.'
    ins_1='Here are some language rules:\n If the two sentences can be identified as similar, then the subjects, verbs and objects of two sentences are similar. Be especially mindful of verb phrases that appear similar but actually have opposite meanings which make sentences dissimilar. '
    #ins_2= 'Determine whether the meaning of sentences can be identified as similar according to the language rules. The answer can only be True or False.  Only output one word.'
    ins_2 = 'Are the following statements similar with the question? Just Say True if they are; otherwise just say False. Only output one word.'

    prompt=ins_0+'\n'+ins_1+'\n'+ins_2+'\n'+evi+'\n'+query

    response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=0.9,
        )
    return response['data']['choices'][0]['content']

def get_llm_chatglm_0(evi,query):
    # your api key
    zhipuai.api_key = " "
    ins='Are the following statements relevant with the question? Just Say True if they are; otherwise just say False. Only output one word.'
    #ins="Are the following sentences similar to the question? For each sentence, if similar, just say true; Otherwise, say false. only output one word. Add the answer to the list and finally just output the list. Here are some examples. \n"
    #examples=" ###Input: \n question: come in \n sentences: ['[1]come out', '[2]come in'] \n ###Response:['False','True'] \n\n\n"
    prompt=ins+'\n'+evi+'\n'+query
    response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=0.9,
        )
    return response['data']['choices'][0]['content']


def call_model_chatgpt(prompt, model, max_tokens=50):
    print(model)
    try:
        results = completions_with_backoff(
            model=model,
            messages=[
                {"role": "user",
                    "content": prompt},
            ],
            request_timeout=60,
            max_tokens=max_tokens,
        )
        result = results["choices"][0]["message"]["content"]
    except (APIError, Timeout, APIConnectionError):
        result = "ERROR: API error outputs"
    return result


def call_model_instructgpt(prompt, model, max_tokens=50):
    try:
        results = completions_instructgpt_backoff(model=model, prompt=prompt, temperature=0.0,
                                                  max_tokens=max_tokens, logprobs=5, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)
        result = results["choices"][0]["text"]
    except (APIError, Timeout, APIConnectionError):
        results = "ERROR: API error outputs"
    return result


def call_model(prompts, model, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)
    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds


def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--input_file', type=str, required=False)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=1,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--query_focus', action="store_true", help='get llm to decide if the query is similar with ctxs.')
    parser.add_argument('--enquery_less', action="store_true", help='reduce the times of getting llm.')
    parser.add_argument('--enquery_file_path', type=str, default=None, help='the file of enquery.')
    parser.add_argument('--rank_file', type=str, default=None, help='the file of rank.')

    args = parser.parse_args()


    args.model_name='../model/selfrag-ll2-7b'
    #args.model_name='ZhipuAI/chatglm3-6b'
    #args.input_file='../data/eval_data/health_claims_processed.jsonl'
    args.input_file = 'src_vmdit/retr_result/HNSW/popqa_longtail.jsonl'
    args.max_new_tokens=100
    args.metric='match'
    args.result_fp='baseline_result/triviaqa'
    args.task='qa'
    args.prompt_name="prompt_no_input_retrieval"
    args.mode='retrieval'
    args.top_n=10
    args.dtype='half'
    args.query_focus=False
    args.enquery_less=True
    args.enquery_file_path = 'src_vmdit/trimmed_evidences/HNSW/trimmed_evidences_popqa_0.json'
    #args.rank_file='./popqa1_hclu_rank.json'


    isOpenAI = True if args.model_name in ["text-davinci-003", "gpt-3.5-turbo-0301", "gpt-3.5-turbo"] else False
    if isOpenAI is False:
        if args.dtype is not None:
            model = LLM(model=args.model_name, download_dir=args.download_dir, dtype=args.dtype,
                        tensor_parallel_size=args.world_size,)
        else:
            model = LLM(model=args.model_name, download_dir=args.download_dir,
                        tensor_parallel_size=args.world_size,trust_remote_code=True)

    input_data = load_file(args.input_file)

    """model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")
    tokenizer_enquery = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_enquery = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
    model_enquery = model_enquery.eval()"""

    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if args.mode == "retrieval":
        l = 20  # 总共检索范围
        n=args.top_n  #检索个数
        mode = 'guass'  # 正态分布
        index_number = get_index_number(l, n,mode)
        if args.retrieval_file is not None:
            retrieval_data = load_file(args.retrieval_file)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if not args.query_focus:
                    item["ctxs"]=item["ctxs"][:args.top_n]
                else:
                    item["ctxs"] = [item["ctxs"][i] for i in index_number]
                if "id" not in item:
                    id2retrieval[id] = item["ctxs"]
                else:
                    id2retrieval[item["id"]] = item["ctxs"]
            i=0
            if not args.enquery_less:
                for id, item in enumerate(input_data):
                    retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                    evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                    if args.query_focus:
                        evidences = llm_evi_ans(model,evidences,item['question'])
                        print("finish:",i)
                        i+=1
                    else:
                        evidences=evidences[:args.top_n]
                    item["paragraph"] = "\n".join(evidences)

            else:
                fin = open(args.enquery_file_path, "r")
                data = fin.read()
                evidences = json.loads(data)
                for id, item in enumerate(input_data):
                    evidence = evidences[id][:args.top_n]
                    item["paragraph"] = "\n".join(evidence)

        else:
            if args.rank_file is None and not args.enquery_less:
                i=0
                for id, item in enumerate(input_data):
                    if not args.query_focus:
                        item["ctxs"]=item["ctxs"][:args.top_n]
                        retrieval_result = item["ctxs"]
                    else:
                        retrieval_result = [item["ctxs"][i] for i in index_number]
                        #retrieval_result = item["ctxs"][:5]
                        #retrieval_result = [item["ctxs"][0],item["ctxs"][2],item["ctxs"][6],item["ctxs"][12],item["ctxs"][19]]
                    #retrieval_result = item["ctxs"][:args.top_n]
                    evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                    if args.query_focus:
                        evidences = llm_evi_ans(model, evidences, item['question'])
                        print("finish:",i)
                        i+=1
                    else:
                        evidences=evidences[:args.top_n]
                    item["paragraph"] = "\n".join(evidences)
                    #print(item['paragraph'])
            elif args.rank_file is not None and not args.enquery_less:
                fin=open(args.rank_file, "r")
                data=fin.read()
                ctxs = json.loads(data)
                i=0
                for id, item in enumerate(input_data):
                    #print(evidences[id])
                    #print(len(ctxs[id]))
                    evidences = [ctxs[id][i] for i in index_number]
                    if args.query_focus:
                        evidences_llm = llm_evi_ans(model, evidences, item['question'])
                        print("finish:", i)
                        i += 1
                    else:
                        evidences_llm=evidences
                    item["paragraph"] = "\n".join(evidences_llm)
            elif args.rank_file is None and args.enquery_less:
                fin = open(args.enquery_file_path, "r")
                data = fin.read()
                evidences = json.loads(data)
                for id, item in enumerate(input_data):
                    evidence = evidences[id][:args.top_n]
                    item["paragraph"] = "\n".join(evidence)
            else:
                print("wrong !")


    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task=='fever' or args.task=='arc_c':
            item["instruction"]=TASK_INST[args.task]+"\n\n### Input:\n" +item["instruction"]
        #else:
         #   item["instruction"] = preprocess_input_data(item, args.task)



    final_results = []
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]
        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            #item["output"] = pred
            final_results.append(item)

    if len(input_data) % args.batch_size > 0:
        l=len(input_data) // args.batch_size
        batch = input_data[l*args.batch_size:]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]
        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(
                processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            #item["output"] = pred
            final_results.append(item)

    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        else:
            raise NotImplementedError
        item["metric_result"] = metric_result

    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))

    if args.task == "factscore":
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        save_file_jsonl(processed_item, args.result_fp)
    else:
        save_file_jsonl(input_data, args.result_fp)


if __name__ == "__main__":
    main()
