# GT4SD's Language Modeling trainer submodule 

Train Language Models via HuggingFace transformers and PyTorch Lightning.


## Installation

### Requirements

Currently `gt4sd-lm-trainer` relies on:

- python>=3.7,<3.9
- pip>=19.1,<20.3

We are actively working on relaxing these, so stay tuned or help us with this by [contributing](./CONTRIBUTING.md) to the project.

### Conda

The recommended way to install the `gt4sd-lm-trainer` is to create a dedicated conda environment, this will ensure all requirements are satisfied. For CPU:

```sh
git clone https://github.com/GT4SD/gt4sd-lm-trainer.git
cd gt4sd-lm-trainer/
conda env create -f conda.yml
conda activate gt4sd
```

**Note:** by default `gt4sd-lm-trainer` is installed with CPU requirements on linux systems. If you have GPU available, run:

```sh
git clone https://github.com/GT4SD/gt4sd-lm-trainer.git
cd gt4sd-lm-trainer/
conda env create -f conda_gpu.yml
conda activate gt4sd
```

### Development setup & installation

If you would like to contribute to the package, we recommend the following development setup:

```sh
conda env create -f conda.yml
conda activate gt4sd-lm-trainer
# install gt4sd-lm-trainer in editable mode
pip install --no-deps -e .
```



### Perform training via the CLI command

GT4SD provides a trainer client based on the `gt4sd-lm-trainer` CLI command. 
```console
$ gt4sd-lm-trainer --help
usage: gt4sd-trainer [-h] [--configuration_file CONFIGURATION_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --configuration_file CONFIGURATION_FILE
                        Configuration file for the trainining. It can be used
                        to completely by-pass pipeline specific arguments.
                        (default: None)
```

To launch a training you have two options.

You can either specify the path of a configuration file that contains the needed training parameters:

```sh
gt4sd-lm-trainer  --training_pipeline_name ${TRAINING_PIPELINE_NAME} --configuration_file ${CONFIGURATION_FILE}
```

Or you can provide directly the needed parameters as arguments:

```sh
gt4sd-lm-trainer --type mlm --model_name_or_path mlm --training_file /path/to/train_file.jsonl --validation_file /path/to/valid_file.jsonl
```


### Convert PyTorch Lightning checkpoints to HuggingFace model via the CLI command

Once a training pipeline has been run via the `gt4sd-lm-trainer`, it's possible to convert the PyTorch Lightning checkpoint
 to HugginFace model via `gt4sd-pl-to-hf`:

```sh
gt4sd-pl-to-hf --hf_model_path ${HF_MODEL_PATH} --training_type ${TRAINING_TYPE} --model_name_or_path ${MODEL_NAME_OR_PATH} --ckpt {CKPT} --tokenizer_name_or_path {TOKENIZER_NAME_OR_PATH}
```



## References

If you use `gt4sd` in your projects, please consider citing the following:

```bib
@article{manica2022gt4sd,
  title={GT4SD: Generative Toolkit for Scientific Discovery},
  author={Manica, Matteo and Cadow, Joris and Christofidellis, Dimitrios and Dave, Ashish and Born, Jannis and Clarke, Dean and Teukam, Yves Gaetan Nana and Hoffman, Samuel C and Buchan, Matthew and Chenthamarakshan, Vijil and others},
  journal={arXiv preprint arXiv:2207.03928},
  year={2022}
}
```

## License

The `gt4sd` codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
