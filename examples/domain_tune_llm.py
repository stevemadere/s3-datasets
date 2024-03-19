
from continuing_trainer import ContinuingTrainer, logger

import os

if ('AWS_PROFILE' not in os.environ) and not 
        ('AWS_SECRET_ACCESS_KEY' in os.environ and
         'AWS_ACCESS_KEY_ID' in os.environ):
  raise EnvironmentError("AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY required in the environment.")

bucket_name=os.environ.get('DATASET_BUCKET')
data_dir = os.environ.get('DATA_DIR')
output_dir = os.environ.get('OUTPUT_DIR')
if 'HF_MODEL_NAME' in os.environ:
    base_model_name = f"{os.environ['HF_CONTRIBUTOR']}/{os.environ['HF_MODEL_NAME']}"
else:
    base_model_name = 'Mistral/Mistral-7Bv0.1'

logger.info(f'base model is {base_model_name}')

dataset_series= os.environ.get('DATASET_SERIES')


steps_per_round = int(os.environ.get('STEPS_PER_ROUND') or 10000)
max_steps = int(os.environ.get('MAX_STEPS') or 25000*10)

continuing_trainer = ContinuingTrainer(
                      base_model_name = base_model_name,
                      bucket_name = bucket_name,
                      output_dir = output_dir,
                      dataset_series = dataset_series,
                      steps_per_round = steps_per_round,
                      max_steps = max_steps
                      )

continuing_trainer.train()

