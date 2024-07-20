# Toxicity Elicitation with AST

This is the source code for our work `ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Likely Toxic Prompts`.

## Setting Up

### Environment
1. Create a new conda environment
`conda create --name ast`
2. Activate the conda environment
`conda activate ast`
3. Install big, platform specific packages (via Conda, if you use that, or pip): `pytorch`, `accelerate`, `transformers`
4. Install the other requirements
`pip3 install -r requirements.txt`

### Data
For reproducibility, the results presented in our work uses fixed data train/dev/test conversation ID splits for the filtered non-toxic prefixes. Please download them [from the `data` subfolder here](https://drive.google.com/drive/folders/11L4yMBzMoeBQEMgdf46l6h4NsRm2whzK?usp=sharing) and place them into `./data`.

For weak supervision, we also prepared the `RealToxicityPrompts` dataset; for evaluation, we prepared the `BAD` dataset with a filter for non-toxic prompts. These support files are [available here](https://drive.google.com/drive/folders/11L4yMBzMoeBQEMgdf46l6h4NsRm2whzK?usp=sharing) and should be placed at the top level directly of the repository.

## Running the Code

### Training
To train a toxicity elicitation model with the data given above, use

```bash
python main.py
```

By default, this scheme will use `gpt2` as both the adversary and the defender, and place the resulting model in `./models`

Call:

```bash
python main.py --help
```

for all options.

### Evaluation
To evaluate the toxicity elicitation of your model, use

```bash
python main_eval.py ./models/your_weights_dir
```


By default, the evaluation results will be given in `./results` as a JSON.

Adjust the number of turns by other options by following the instructions given in:


```
python main_eval.py --help
```

## Citing the Work
If this was useful, please consider citing:

```
@misc{hardy2024astprompter,
  title={ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Likely Toxic Prompts},
  author={Hardy, Amelia F and Liu, Houjun and Lange, Bernard and Kochenderfer, Mykel J},
  journal={arXiv preprint arXiv:2407.09447},
  year={2024}
}
```

If you run into any issues, please feel free to email `{houjun,ahardy} at stanford dot edu`.
