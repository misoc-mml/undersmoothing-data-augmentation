ITER=1000
simple=("dummy" "ridge" "lasso" "kr" "dt" "lgbm" "et" "cb" "cf")
meta=("dml" "tl" "xl" "tb")
base=("ridge" "lasso" "kr" "dt" "lgbm" "et" "cb")

bash ./build_results_dirs.sh .. main_results

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/main_results/ihdp_${MODEL} --sr --sfi --tbv --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}-${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/main_results/ihdp_${MODEL}-${BASE_MODEL} --sr --sfi --tbv --em $MODEL --ebm $BASE_MODEL
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results/main_results --dtype ihdp -o ../results/main_results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show


bash ./build_results_dirs.sh .. rules_count

for MODEL in "dt"
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/rules_count/ihdp_${MODEL} --sr --sfi --tbv --em $MODEL --rules
done

for MODEL in "tb"
do
    for BASE_MODEL in "dt"
    do
        echo ${MODEL}-${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/rules_count/ihdp_${MODEL}-${BASE_MODEL} --sr --sfi --tbv --em $MODEL --ebm $BASE_MODEL --rules
    done
done