PYTHON="/home/li/.conda/envs/neurosim_test/bin/python"
############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

num_runs=6
model=resnet50

dataset=cubs_cropped
data_path='/home/li/data/imagenet_to_sketch/cubs_cropped'
# data_path='/opt/imagenet/imagenet_compressed/'

epochs=30
batch_size=64
optimizer=SGD

# quantization scheme
channel_wise=0
wbit=4
abit=4

wd=1e-4
lr=5e-3

pretrained_model="./save/resnet50/resnet50_lr0.005_wd0.0001_channelwise0/model_best.pth.tar"

for run_id in `seq 1 $num_runs`;
do
    save_path="./save/${model}_CL/${dataset}/${model}_lr${lr}_wd${wd}_channelwise${channel_wise}/"
    log_file="${model}_lr${lr}_wd${wd}_wbit${wbit}_abit${abit}_run${run_id}.log"

    $PYTHON -W ignore train.py --dataset ${dataset} \
        --data_path ${data_path} \
        --model ${model} \
        --save_path ${save_path} \
        --epochs ${epochs} \
        --log_file ${log_file} \
        --lr  ${lr} \
        --schedule 15 \
        --gammas 0.1 \
        --batch_size ${batch_size} \
        --ngpu 4 \
        --wd ${wd} \
        --wbit ${wbit} \
        --abit ${abit} \
        --channel_wise ${channel_wise} \
        --fine_tune \
        --resume ${pretrained_model};
done
    
