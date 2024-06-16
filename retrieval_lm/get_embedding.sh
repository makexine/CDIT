python3 generate_passage_embeddings.py \
        --model_name_or_path ../model/contriever-msmarco \
        --passages ./retr_result/popqa1.jsonl \
        --output_dir ex_embeddings_hclu \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500
