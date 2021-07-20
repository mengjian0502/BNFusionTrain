PYTHON="/home/li/.conda/envs/neurosim_test/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet50

dataset=imagenet
data_path='/opt/imagenet/imagenet_compressed/'

epochs=60
batch_size=128
optimizer=SGD

# quantization scheme
channel_wise=0
wbit=4
abit=4

wd=0.0001
lr=0.01

save_path="./save/${model}/${model}_lr${lr}_wd${wd}_channelwise${channel_wise}/"
log_file="${model}_lr${lr}_wd${wd}_wbit${wbit}_abit${abit}.log"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ${data_path} \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 30 45 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 4 \
    --wd ${wd} \
    --wbit ${wbit} \
    --abit ${abit} \
    --channel_wise ${channel_wise} \
    --fine_tune \
    --pretrained;

    
