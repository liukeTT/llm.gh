/*
This code is a convenience tool for profiling the CUDA kernels in the training
loop of train_gpt2.cu. Compile:

make profile_gpt2cu NO_MULTI_GPU=1

And then e.g. use ncu from NVIDIA. The CLI docs for example:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/

TLDR run like:

sudo ncu --set full --import-source yes -o profile -f ./profile_gpt2cu

This:
- `--set full` means we'll collect A LOT of metrics. take out for less
- `--import-source yes` means we'll get the source code in the profile
- `-o profile` writes the results into file profile.ncu-rep
- `-f` forces overwrite of the profile.ncu-rep file
- `./profile_gpt2cu` is the executable we want to profile

This writes results into profile.ncu-rep output file.
You can open this up in NVIDIA Nsight Compute UI.
For example, I have NVIDIA Nsight Compute installed on my Mac, and I rsync
the profile.ncu-rep from a cloud box to local to pretty view.
*/

#define TESTING
#include "train_gpt2.cu"


void gpt2_build_from_size(GPT2 *model, int size, int B, int T) {    
    int depth, channels, num_heads;
    if      (size == 7)  { depth = 36; channels = 4096; num_heads = 32; } // 8B
    else if (size == 70) { depth = 60; channels = 10240; num_heads = 80; } // 70B
    else if (size == 145) { depth = 80; channels = 12288; num_heads = 96; } // 145B
    else if (size == 300) { depth = 96; channels = 16384; num_heads = 128; } // 300B
    else if (size == 1000) { depth = 128; channels = 25600; num_heads = 160; } // 1T
    else { fprintf(stderr, "Unsupported size for now\n"); exit(EXIT_FAILURE); }

    model->config.num_layers = depth;
    model->config.channels = channels;
    model->config.num_heads = num_heads;
    model->config.max_seq_len = 2048;
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128

    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
    }
    
    float m = (float)model->num_parameters * 4 / (1024 * 1024 * 1024);
    float v = (float)model->num_parameters * 4 / (1024 * 1024 * 1024);
    float p = (float)model->num_parameters * 2 / (1024 * 1024 * 1024);
    float g = (float)model->num_parameters * 2 / (1024 * 1024 * 1024);
    printf("allocating %f GiB for m\n", m);
    printf("allocating %f GiB for v\n", v);
    printf("allocating %f GiB for p\n", p);
    printf("allocating %f GiB for g\n", g);

    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type);
    }
    float a = (float)bytes / (1024 * 1024 * 1024);
    printf("allocating %f GiB for activations\n", a);
    
    printf("m+v consumes %f %% of memory space\n", (m + v) / (a + m + v + p + g) * 100);
    printf("p+g consumes %f %% of memory space\n", (p + g) / (a + m + v + p + g) * 100);
    printf("activation consumes %f %% of memory space\n", a / (a + m + v + p + g) * 100);
}


int main(int argc, char *argv[]) {
    int ModelSize[5] = {7, 70, 145, 300, 1000};
    int BatchSize[8] = {1, 4, 8, 16, 32, 64, 128, 256};

    for (int i = 0; i < 5; i++) {
        printf("\n");
        int model_size = ModelSize[i];
        for (int j = 0; j < 8; j++) {
            int B = BatchSize[j];
            int T = 2048;

            printf("-----------------\n");
            printf("GPT-%dB, BatchSize %d, Sequence Length %d\n", model_size, B, T);

            // build the GPT-2 model by model size
            GPT2 model;
            gpt2_init_common(&model);
            gpt2_build_from_size(&model, model_size, B, T);
        }
    }
    return 0;
}
