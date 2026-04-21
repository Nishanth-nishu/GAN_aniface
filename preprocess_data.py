import sys
from data.preprocessor import DataPreprocessor
p = DataPreprocessor("/scratch/nishanth.r/gan_proj/data_raw", "/scratch/nishanth.r/gan_proj/data_processed")
p.run_pipeline()
