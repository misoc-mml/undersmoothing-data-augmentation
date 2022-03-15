ITER=10
simple=("kr" "dt")
meta=("xl" "tb")
base=("kr" "dt")

bash ./build_results_dirs.sh .. demo_results

# Standalone estimators
for MODEL in ${simple[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/demo_results/ihdp_${MODEL} --sr --sfi --tbv --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}-${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/demo_results/ihdp_${MODEL}-${BASE_MODEL} --sr --sfi --tbv --em $MODEL --ebm $BASE_MODEL
    done
done

# Post-processing the results
python ../results/process.py --data_path ../results/demo_results --dtype ihdp -o ../results/demo_results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show