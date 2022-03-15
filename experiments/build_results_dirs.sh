# Assume all results must be stored under the main 'results' directory.
results_path="$1/results"
if [ ! -d $results_path ]
then
    mkdir $results_path
fi

# Create a desired nested directory if necessary.
nested_path="$results_path/$2"
if [ ! -d $nested_path ]
then
    mkdir $nested_path
fi