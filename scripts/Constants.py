# Data Loading
expr_path="C:/Users/solam/Desktop/METABRIC/data/raw/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
clinical_path="C:/Users/solam/Desktop/METABRIC/data/raw/data_clinical_patient.txt"
relevant_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']

# Pre processing
zscore=True
pca_components=None


# Train/Test split parameters
batchsize = 64
valtestsize = 0.3  # 30% of the data will be used for validation + test
testsize = 0.5     # Of the 30% allocated to val+test, half will be for validation and half for test → 15% of total data each
