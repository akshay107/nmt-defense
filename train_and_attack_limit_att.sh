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
size=${11}

bash tools/bpe_pipeline_src_tgt_limit_att.sh $src $tgt $epoch_start $emb_topk $lambda $clip $frac_replace $context $topk $gpu


if python get_good_words.py $src $tgt ./temp/run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}/data/
then
    echo "check successful for run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}"
else
    exit
fi

cp ./temp/run_${src}_${tgt}/test.src.reduced.final ./temp/run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}/


# run bruteforce attack

echo "running bruteforce attack for run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}"

folder="run_${src}_${tgt}_ep_${epoch_start}_emb_${emb_topk}_lambda_${lambda}_clip_${clip}_frac_${frac_replace}_con_${context}_top_${topk}"
out_file="bf_output.txt"

python translate.py -i temp/$folder/data/ --data processed --output temp/$folder/test.out.bpe.reduced.final --best_model_file temp/$folder/models/model_best_$folder.ckpt --src temp/$folder/test.src.reduced.final --gpu 0

cat temp/$folder/test.out.bpe.reduced.final | sed -E 's/(@@ )|(@@ ?$)//g' > temp/$folder/test.out.reduced.final

bash tools/bruteforce_bpe_pipeline_src_tgt_limit_att.sh $src $tgt $context $topk $folder $out_file $gpu $size

bash get_output.sh ${out_file%.*} $folder
