# decode and encode since it can lead to different subword units
file=$1
folder=$2

OUT=temp/$folder/output

cat $OUT/$1.txt | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/$1.txt.decoded
./bin/apply_bpe -c ./temp/$folder/data/bpe-codes.32000 < $OUT/$1.txt.decoded > $OUT/$1.txt.encoded

python translate.py -i temp/$folder/data/ --data processed --output $OUT/$1.out --best_model_file temp/$folder/models/model_best_$folder.ckpt --src $OUT/$1.txt.encoded --gpu 0

mv $OUT/$1.out{,.bpe}

cat $OUT/$1.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/$1.out.decoded


#python get_stats.py ./temp/$folder/test.src.reduced.final $OUT/$1.txt.encoded ./temp/$folder/test.out.reduced.final $OUT/$1.out.decoded > temp/$folder/output/report.txt

python get_stats.py ./temp/$folder/test.src.reduced.final $OUT/$1.txt ./temp/$folder/test.out.reduced.final $OUT/$1.out.decoded > temp/$folder/output/report.txt
