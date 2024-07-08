# GPT-2 (124M) repro on FineWeb
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

# watch -n 0.1 nvidia-smi

# dataset preprocessing
# python dev/data/tinyshakespeare.py
train_data="dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
val_data="dev/data/tinyshakespeare/tiny_shakespeare_val.bin"

train_data="dev/data/fineweb10B/fineweb_train_000001.bin"
val_data="dev/data/fineweb10B/fineweb_val_000000.bin"

make train_gpt2cu_gh USE_CUDNN=1 NO_MULTI_GPU=1

max_steps=60
batch_size=256
seq_len=1024

out_dir="log_gpt2"
mkdir -p ${out_dir}
rm -fr ${out_dir}/*

gpt="d12"
model_size="124M"
for batch_size in 4 8 16 32 64 128; do
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 0 -w 0 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p16
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 1 -w 0 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p16.mv_offload

    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 0 -w 1 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p32
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 1 -w 1 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p32.mv_offload
done

gpt="d24"
model_size="350M"
for batch_size in 4 8 16 32 64; do
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 0 -w 0 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p16
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 1 -w 0 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p16.mv_offload

    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 0 -w 1 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p32
    ./train_gpt2cu_gh -i ${train_data} -j ${val_data} -x ${max_steps} -b ${batch_size} -t ${seq_len} -e ${gpt} -zm 1 -w 1 2>&1 | tee $out_dir/${model_size}.bs${batch_size}.p32.mv_offload
done