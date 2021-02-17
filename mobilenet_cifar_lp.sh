PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=mobilenetv1_Q
dataset=cifar10
batch_size=128
optimizer=SGD

channel_wise=0
w_profit=4

wd=0.00004
lr=0.0005

save_path="./save/${model}/${model}_lr${lr}_wd${wd}_PROFIT/"
log_file="${model}_lr${lr}_wd${wd}.log"

pretrained_model="./pretrained/MaskedMobileNetV1_cifar_FP_cifar10/model_best.pth"

$PYTHON -W ignore train_profit.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --w_bit 8 5 4 \
    --a_bit 8 5 4 \
    --w_profit ${w_profit} \
    --resume ${pretrained_model} \
    --fine_tune \
    --stabilize \
    --teacher self \



    
