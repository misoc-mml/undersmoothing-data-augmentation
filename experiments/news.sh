ITER=50
# No ET and CB due to extremely long training time on this data set.
simple=("dummy" "lr" "lr-ipw" "ridge" "ridge-ipw" "lasso" "kr" "kr-ipw" "dt" "dt-ipw" "lgbm" "lgbm-ipw" "cf")
meta=("dml" "dr" "tl" "xl")
base=("lr" "ridge" "lasso" "kr" "dt" "lgbm")

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/news_${MODEL} --sr --tbv --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}_${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/NEWS --dtype news --iter $ITER -o ../results/news_${MODEL}-${BASE_MODEL} --sr --tbv --em $MODEL --ebm $BASE_MODEL
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results --dtype news -o ../results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show