# docker build -t llm -f ./Dockerfile .
# docker build -t llm_cudnn_arm -f ./Dockerfile_cudnn_arm .
# docker build -t llm_cudnn_x86 -f ./Dockerfile_cudnn_x86 .
# docker system prune --volumes

docker_img="llm_cudnn_arm:latest"
#docker_img="llm_cudnn_x86:latest"

home_dir_0="${HOME}/llm.gh"
docker_dir_0="/workspace/llm.gh"

work_dir=${docker_dir_0}

docker run --gpus all --rm -it -P \
    --cap-add=SYS_ADMIN \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${home_dir_0}:${docker_dir_0} \
    -w /${work_dir} \
    ${docker_img} \
    bash
