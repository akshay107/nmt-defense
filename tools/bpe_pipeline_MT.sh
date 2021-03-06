#!/usr/bin/env bash

TF=$(pwd)

export PATH=$PATH:$TF/bin
#======= EXPERIMENT SETUP ======

L1=$1
L2=$2
L3=$3

share_sublayer=$4
# options are: q,k,v,f,linear

attn_share=$5
# options are: self, source

prefix="MT_$4_$5"

# update these variables
NAME="run_${prefix}_${L1}_${L2}-${L3}"
OUT="temp/$NAME"

DATA=${TF}"/data/${L1}_${L2}-${L3}"
TRAIN_SRC=$DATA/train.${L1}
TRAIN_TGT=$DATA/train.${L2}-${L3}
TEST_SRC=$DATA/test.${L1}
TEST_TGT=$DATA/test.${L2}-${L3}
VALID_SRC=$DATA/dev.${L1}
VALID_TGT=$DATA/dev.${L2}-${L3}

BPE_OPS=32000
GPUARG=0

#====== EXPERIMENT BEGIN ======


echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test

echo "Step 1a: Preprocess inputs"

echo "Learning BPE on source and target combined"
cat ${TRAIN_SRC} ${TRAIN_TGT} | learn_bpe -s ${BPE_OPS} > $OUT/data/bpe-codes.${BPE_OPS}

echo "Applying BPE on source"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TRAIN_SRC > $OUT/data/train.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $VALID_SRC > $OUT/data/valid.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TEST_SRC > $OUT/data/test.src

echo "Applying BPE on target"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_TGT > $OUT/data/train.tgt
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_TGT > $OUT/data/valid.tgt
# We dont touch the test References, No BPE on them!
cp $TEST_TGT $OUT/data/test.tgt


echo "Step 1b: Preprocess"
python ${TF}/preprocess.py -i ${OUT}/data \
      -s-train train.src \
      -t-train train.tgt \
      -s-valid valid.src \
      -t-valid valid.tgt \
      -s-test test.src \
      -t-test test.tgt \
      --save_data processed \
      --max_seq_len 70


echo "Step 2: Train"
CMD="python $TF/train.py -i $OUT/data --data processed \
--model_file $OUT/models/model_$NAME.ckpt --best_model_file $OUT/models/model_best_$NAME.ckpt \
--data processed --batchsize 30 --tied --beam_size 5 --epoch 30 \
--layers 6 --multi_heads 8 --gpu $GPUARG \
--dev_hyp $OUT/test/valid.out --test_hyp $OUT/test/test.out \
--model MultiTaskNMT --metric bleu --wbatchsize 2000 --max_decode_len 70 \
--lang1 __${L2}__ --lang2 __${L3}__ \
--pshare_decoder_param --grad_accumulator_count 2 --share_sublayer ${share_sublayer} --attn_share ${attn_share}"

echo "Training command :: $CMD"
eval "$CMD"

bash tools/bpe_translate_O2M.sh ${L1} ${L2} ${L3} MultiTaskNMT ${prefix}
# bash tools/bpe_translate_O2M.sh en es pt MultiTaskNMT MT