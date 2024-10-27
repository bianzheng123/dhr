#git clone --progress https://huggingface.co/jacklin/DistilBERT-AGG
export MODEL_DIR=DistilBERT-AGG
export CUDA_VISIBLE_DEVICES=0
export MODEL=AGG
export AGGDIM=640
export CORPUS=scifact
python -m tevatron.datasets.beir.encode_and_retrieval --dataset ${CORPUS} --model_name_or_path ${MODEL_DIR} --model ${MODEL} --agg_dim ${AGGDIM} 