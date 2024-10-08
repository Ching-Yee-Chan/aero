import math
import os

import time

import hydra
import torch
import logging
from pathlib import Path

import torchaudio
from torchaudio.functional import resample

from src.enhance import write
from src.models import modelFactory
from src.model_serializer import SERIALIZE_KEY_MODELS, SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_STATE
from src.utils import bold

logger = logging.getLogger(__name__)


SEGMENT_DURATION_SEC = 10

def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)['generator']
    package = torch.load(checkpoint_file, 'cpu')
    load_best = args.continue_best
    if load_best:
        logger.info(bold(f'Loading model {model_name} from best state.'))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])
    else:
        logger.info(bold(f'Loading model {model_name} from last state.'))
        model.load_state_dict(package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])

    return model


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    print(args)
    model = _load_model(args)

    rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    rank = (rank % max_gpu)
    device = torch.device(f"cuda:{rank}")

    model.to(device)
    audio_list_file = args.il
    audio_list = []
    with open(audio_list_file, 'r') as f:
        for line in f:
            audio_list.append(line.strip())
    
    for relative_path in audio_list:
        # file_basename = Path(filename).stem
        absolute_path = os.path.join(args.audio_data_dir, relative_path)
        output_dir = args.output
        lr_sig, sr = torchaudio.load(absolute_path)

        out_filename = os.path.join(output_dir, relative_path)
        if os.path.exists(out_filename):
            logger.info(f'{out_filename} already exists. Skipping...')
            continue
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)

        if args.experiment.upsample:
            lr_sig = resample(lr_sig, sr, args.experiment.hr_sr)
            sr = args.experiment.hr_sr

        logger.info(f'lr wav shape: {lr_sig.shape}')

        segment_duration_sec = SEGMENT_DURATION_SEC
        while True:
            segment_duration_samples = sr * segment_duration_sec
            n_chunks = math.ceil(lr_sig.shape[-1] / segment_duration_samples)

            lr_chunks = []
            for i in range(n_chunks):
                start = i * segment_duration_samples
                end = min((i + 1) * segment_duration_samples, lr_sig.shape[-1])
                lr_chunks.append(lr_sig[:, start:end])

            if lr_chunks[-1].shape[-1] > 256:
                break
            elif segment_duration_sec == 1:
                raise ValueError('Segment duration is too small')
            else:
                segment_duration_sec -= 1
                logger.info(f'Segment duration reset to {segment_duration_sec} seconds')
        
        logger.info(f'number of chunks: {n_chunks}')

        pr_chunks = []

        model.eval()
        pred_start = time.time()
        with torch.no_grad():
            for i, lr_chunk in enumerate(lr_chunks):
                pr_chunk = model(lr_chunk.unsqueeze(0).to(device)).squeeze(0)
                logger.info(f'lr chunk {i} shape: {lr_chunk.shape}')
                logger.info(f'pr chunk {i} shape: {pr_chunk.shape}')
                pr_chunks.append(pr_chunk.cpu())

        pred_duration = time.time() - pred_start
        logger.info(f'prediction duration: {pred_duration}')

        pr = torch.concat(pr_chunks, dim=-1)

        logger.info(f'pr wav shape: {pr.shape}')

        logger.info(f'saving to: {out_filename}, with sample_rate: {args.experiment.hr_sr}')
        write(pr, out_filename, args.experiment.hr_sr)

"""
Need to add filename and output to args.
Usage: python predict.py <dset> <experiment> +filename=<path to input file> +output=<path to output dir>
"""
if __name__ == "__main__":
    main()