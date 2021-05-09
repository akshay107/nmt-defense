#!/usr/bin/env bash
src=$1
tgt=$2
epoch_start=$3
emb_topk=$4
lambda=$5
clip=$6
frac_replace=$7
context=$8
topk=$9
gpu=${10}
TF=$(pwd)
gpu=${gpu/,/ }
export PATH=$TF/bin:$PATH
#======= EXPERIMENT SETUP ======

# update these variables
NAME="run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}"
OUT="temp/$NAME"
# OUT="temp/$NAME"

DATA=${TF}"/data/${src}_${tgt}"
TRAIN_SRC=$DATA/train.$src
TRAIN_TGT=$DATA/train.$tgt
TEST_SRC=$DATA/test.$src
TEST_TGT=$DATA/test.$tgt
VALID_SRC=$DATA/dev.$src
VALID_TGT=$DATA/dev.$tgt

BPE="src+tgt" # src, tgt, src+tgt
BPE_OPS=32000
#GPUARG="0"

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

python $TF/get_good_words.py $src $tgt $OUT/data/
echo "Step 2: Train"
if [ ${#gpu} -eq 1 ]
then
CMD="python $TF/train_new.py -i $OUT/data --data processed \
--model_file $OUT/models/model_$NAME.ckpt --best_model_file $OUT/models/model_best_$NAME.ckpt \
--batchsize 30 --tied --beam_size 5 --epoch 15 \
--layers 6 --multi_heads 8 --gpu $gpu --max_decode_len 70 \
--dev_hyp $OUT/test/valid.out --test_hyp $OUT/test/test.out \
--model Transformer --metric bleu --wbatchsize 3000 --log_path $OUT/${NAME}.log --lambda_val $lambda --clip $clip --frac_replace $frac_replace --context $context --topk $topk --epoch_start $epoch_start --emb_topk $emb_topk"
else
CMD="python $TF/train_new.py -i $OUT/data --data processed \
--model_file $OUT/models/model_$NAME.ckpt --best_model_file $OUT/models/model_best_$NAME.ckpt \
--batchsize 30 --tied --beam_size 5 --epoch 15 \
--layers 6 --multi_heads 8 --multi_gpu $gpu --gpu 0 --max_decode_len 70 \
--dev_hyp $OUT/test/valid.out --test_hyp $OUT/test/test.out \
--model Transformer --metric bleu --wbatchsize 3000 --log_path $OUT/${NAME}.log --lambda_val $lambda --clip $clip --frac_replace $frac_replace --context $context --topk $topk --epoch_start $epoch_start --emb_topk $emb_topk"
fi
echo "Training command :: $CMD"
eval "$CMD"

echo "BPE decoding/detokenising target to match with references"
mv $OUT/test/test.out{,.bpe}
mv $OUT/test/valid.out{,.bpe}
cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

echo "Step 4a: Evaluate Test"
perl $TF/tools/multi-bleu.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
perl $TF/tools/multi-bleu.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

echo "Test Bleu Score" >> $OUT/${NAME}.log
t2t-bleu --translation=$OUT/test/test.out --reference=$OUT/data/test.tgt >> $OUT/${NAME}.log
echo "" >> $OUT/${NAME}.log

echo "EMA Test Bleu score" >> $OUT/${NAME}.log
mv $OUT/test/test.out.ema $OUT/test/test.out.ema.bpe
mv $OUT/test/valid.out.ema $OUT/test/valid.out.ema.bpe
cat $OUT/test/valid.out.ema.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out.ema
cat $OUT/test/test.out.ema.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out.ema

t2t-bleu --translation=$OUT/test/test.out.ema --reference=$OUT/data/test.tgt >> $OUT/${NAME}.log

echo "Value of epoch_start is $epoch_start" >> $OUT/${NAME}.log
echo "Value of emb_topk is $emb_topk" >> $OUT/${NAME}.log
echo "Value of lambda is $lambda" >> $OUT/${NAME}.log
echo "Value of clip is $clip" >> $OUT/${NAME}.log
echo "Value of frac_replace is $frac_replace" >> $OUT/${NAME}.log
echo "Value of context is $context" >> $OUT/${NAME}.log
echo "Value of topk is $topk" >> $OUT/${NAME}.log
