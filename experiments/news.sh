ITER=50
# No ET and CB due to extremely long training time on this data set.
simple=("dummy" "ridge" "lasso" "kr" "dt" "lgbm" "cf")
meta=("dml" "tl" "xl" "tb")
base=("ridge" "lasso" "kr" "dt" "lgbm")

bash ./build_results_dirs.sh .. main_results

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/main_results/news_${MODEL} --sr --sfi --tbv --em $MODEL --dt_md 20
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}-${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/main_results/news_${MODEL}-${BASE_MODEL} --sr --sfi --tbv --em $MODEL --ebm $BASE_MODEL --cov_type diag --dt_md 20
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results/main_results --dtype news -o ../results/main_results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show


bash ./build_results_dirs.sh .. rules_count

for MODEL in "dt"
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/rules_count/news_${MODEL} --sr --sfi --tbv --em $MODEL --dt_md 20 --rules
done

for MODEL in "tb"
do
    for BASE_MODEL in "dt"
    do
        echo ${MODEL}-${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/rules_count/news_${MODEL}-${BASE_MODEL} --sr --sfi --tbv --em $MODEL --ebm $BASE_MODEL --cov_type diag --dt_md 20 --rules
    done
done