ITER=10
simple=("dummy" "ridge-ipw" "lasso" "kr-ipw" "lgbm")
meta=("dml" "tl")
base=("ridge" "lasso")

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/ihdp_${MODEL} --sr --tbv --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}_${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/ihdp_${MODEL}-${BASE_MODEL} --sr --tbv --em $MODEL --ebm $BASE_MODEL
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results --dtype ihdp -o ../results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show