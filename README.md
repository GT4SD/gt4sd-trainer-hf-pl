# GT4SD's trainer submodule for HF transformers and PyTorch Lightning

Train Language Models via HuggingFace transformers and PyTorch Lightning.


### Development setup & installation

Create any virtual or conda environment compatible with the specs in setup.cfg. Then run:
```sh
pip install -e ".[dev]" 
```



### Perform training via the CLI command

GT4SD provides a trainer client based on the `gt4sd-lm-trainer` CLI command. 
```console
$ gt4sd-trainer-lm --help
usage: gt4sd-trainer-lm [-h] [--configuration_file CONFIGURATION_FILE]

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
gt4sd-trainer-lm  --training_pipeline_name ${TRAINING_PIPELINE_NAME} --configuration_file ${CONFIGURATION_FILE}
```

Or you can provide directly the needed parameters as arguments:

```sh
gt4sd-trainer-lm --type mlm --model_name_or_path mlm --training_file /path/to/train_file.jsonl --validation_file /path/to/valid_file.jsonl
```


### Convert PyTorch Lightning checkpoints to HuggingFace model via the CLI command

Once a training pipeline has been run via the `gt4sd-lm-trainer`, it's possible to convert the PyTorch Lightning checkpoint
 to HugginFace model via `gt4sd-pl-to-hf`:

```sh
gt4sd-pl-to-hf --hf_model_path ${HF_MODEL_PATH} --training_type ${TRAINING_TYPE} --model_name_or_path ${MODEL_NAME_OR_PATH} --ckpt {CKPT} --tokenizer_name_or_path {TOKENIZER_NAME_OR_PATH}
```



### References

If you use `gt4sd` in your projects, please consider citing the following:

```bib
@article{manica2022gt4sd,
  title={GT4SD: Generative Toolkit for Scientific Discovery},
  author={Manica, Matteo and Cadow, Joris and Christofidellis, Dimitrios and Dave, Ashish and Born, Jannis and Clarke, Dean and Teukam, Yves Gaetan Nana and Hoffman, Samuel C and Buchan, Matthew and Chenthamarakshan, Vijil and others},
  journal={arXiv preprint arXiv:2207.03928},
  year={2022}
}
```

### License

The `gt4sd` codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
