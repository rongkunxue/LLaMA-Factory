export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118


cd /opt/tiger

cp -r /mnt/bn/valley2/liuyuhang/repos/SafetyGPTOmniBenchmarkEval/libs/transformers /opt/tiger/
pip3 install /opt/tiger/transformers

sudo pip3 uninstall torch torchvision torchaudio -y
sudo pip3 uninstall deepspeed -y
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

cd /mnt/bn/evalonly/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install --no-deps -e .
pip3 install trl==0.9.6
pip3 install tyro==0.8.14
sudo pip3 uninstall -y gradio

pip3 install qwen-vl-utils[decord]==0.0.10
pip3 install flash-attn==2.7.0.post2 --no-build-isolation
DS_BUILD_CPU_ADAM=1 pip3 install deepspeed==0.14.4

pip3 install accelerate==0.34.0
pip3 install liger_kernel
#cp -r /mnt/bn/valley2/xrk/train_dpo.yaml /mnt/bn/evalonly/LLaMA-Factory/examples/train_lora/a.yaml
#cp -r /mnt/bn/gne-distill-r1/xrk_result/dpo_grpo_test_smaple_dpo/scoreresult_dpo.json /mnt/bn/evalonly/LLaMA-Factory/data/scoreresult_dpo.json
#cp -r /opt/tiger/EasyR1/DPO/LLaMA-Factory/data/dataset_info.json /mnt/bn/evalonly/LLaMA-Factory/data/dataset_info.json
#llamafactory-cli train /opt/tiger/EasyR1/DPO/LLaMA-Factory/examples/train_lora/a.yaml

