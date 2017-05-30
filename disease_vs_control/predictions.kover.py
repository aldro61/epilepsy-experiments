"""
Set Covering Machine (Kover) predictions for each train/test split

"""
import os

from kover.dataset import KoverDataset


n_cpu = 4
phenotype_name = "Epilepsy case vs control"
genomic_data_path = "/exec5/GROUP/pacoss/COMMUN/claudia/machine_learning/data/ibd_clusters.kmers.matrix"
phenotype_data_path = "/exec5/GROUP/pacoss/COMMUN/claudia/machine_learning/data/ibd_clusters.pheno.txt"
kover_dataset_path = "./data/dataset.kover"  #"/exec5/GROUP/pacoss/COMMUN/adrouin/epilepsie/data/disease_vs_control/dataset.kover"


# Create the Kover dataset if needed
if not os.path.exists(kover_dataset_path):
    os.system(
"""
kover dataset create from-tsv \
    --genomic-data {0!s} \
    --phenotype-metadata {1!s} \
    --phenotype-name "{2!s}" \
    --output {3!s} \
    -v
""".format(genomic_data_path, phenotype_data_path, phenotype_name, kover_dataset_path))

# Create the splits if needed
dataset = KoverDataset(kover_dataset_path)
for split in os.listdir("splits"):
    if split not in [s.name for s in dataset.splits]:
        train_ids = [l.strip() for l in open(os.path.join("splits", split, "train_ids.tsv"), "r")]
        test_ids = [l.strip() for l in open(os.path.join("splits", split, "test_ids.tsv"), "r")]

        os.system(
"""
kover dataset split \
    --dataset {0!s} \
    --id {1!s} \
    --train-ids {2!s} \
    --test-ids {3!s} \
    --folds 10 \
    --random-seed 42 \
    -v
""".format(kover_dataset_path, split, " ".join(train_ids), " ".join(test_ids)))

# Run the experiments
if not os.path.exists(os.path.join("predictions", "kover.scm")):
    os.mkdir(os.path.join("predictions", "kover.scm"))

for split in os.listdir("splits"):
    output_dir = os.path.join("predictions", "kover.scm", split)

    if not os.path.exists(os.path.join(output_dir, "results.json")):
        os.system(
"""
kover learn \
    --dataset {0!s} \
    --split {1!s} \
    --model-type conjunction disjunction \
    --p 1.0 1.4678 2.1544 3.1623 4.6416 6.8129 10.0 14.678 21.5443 31.6228 \
    --max-rules 30 \
    --hp-choice cv \
    --random-seed 42 \
    --n-cpu {2:d} \
    --output-dir {3!s} \
    -x
""".format(kover_dataset_path, split, n_cpu, output_dir))

        # TODO: Save the predictions in the same format as the sklearn algorithms

