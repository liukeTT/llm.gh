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

make train_gpt2cu_gh USE_CUDNN=1 NO_MULTI_GPU=1

#./train_gpt2cu_gh -i ${train_data} -j ${val_data} -e "d12" -w 0
./train_gpt2cu_gh -i ${train_data} -j ${val_data} -e "d12" -zm 1 -w 0

#./train_gpt2cu_gh -i ${train_data} -j ${val_data} -e "d12"
#./train_gpt2cu_gh -i ${train_data} -j ${val_data} -e "d12" -zm 1
