src=$1
tgt=$2
gpu=$3
size=$4

bash tools/bpe_pipeline_src_tgt.sh $src $tgt $gpu

if python get_good_words.py $src $tgt ./temp/run_${src}_${tgt}/data/
then
    echo "check successful for run_${src}_${tgt}"
else
    exit
fi

# sanity check passed. now prepare test.src.reduced.final (same for both).

awk 'NF<=40' ./temp/run_${src}_${tgt}/data/test.src | awk 'NF>5' | shuf -n 500 > ./temp/run_${src}_${tgt}/test.src.reduced.final

# run bruteforce attack

echo "running bruteforce attack for run_${src}_${tgt}"

folder="run_${src}_${tgt}"
out_file="bf_output.txt"

python translate.py -i temp/$folder/data/ --data processed --output temp/$folder/test.out.bpe.reduced.final --best_model_file temp/$folder/models/model_best_$folder.ckpt --src temp/$folder/test.src.reduced.final --gpu 0

cat temp/$folder/test.out.bpe.reduced.final | sed -E 's/(@@ )|(@@ ?$)//g' > temp/$folder/test.out.reduced.final

bash tools/bruteforce_bpe_pipeline_src_tgt.sh $src $tgt $folder $out_file $gpu $size
bash get_output.sh ${out_file%.*} $folder
