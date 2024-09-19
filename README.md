# CDIT： Context-Driven Index Trimming
An approach from data quality perspective to enhance the answer of RAG.

### Knowledge preparation
```bash
cd retrieval_lm
```
First, to get the embeddings of knowledge base:
```bash
python3 generate_passage_embeddings.py \
        --model_name_or_path ../model/contriever-msmarco \   #pre-trained retriever
        --passages  \      #knowldge base
        --output_dir ex_embeddings \ 
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500
```
or
```bash
bash get_embedding,sh
```

Retrieve related passages with query
```bash
python passage_retrieve.py
```

### Baseline RAG

LLM generate answers
```bash
python run_baseline_lm.py
```


### CDIT
LLM generate answers
```bash
python run_baseline_lm.py \
       -- enquery_less True \
       -- enquery_file_path \
```
