# Dialogue Toxicity Elicitation with AST

## Set up instructions
1. Create a new conda environment
`conda create --name detox`
2. Activate the conda environment
`conda activate detox`
3. Install big, platform specific packages (via Conda, if you use that, or pip): `pytorch`, `accelerate`, `transformers`, `trl`
4. Install the other requirements
`pip3 install -r requirements.txt`
5. Set up .env file with Perspective credential
`echo PERSPECTIVE_API_KEY="insert your key here" > .env` 
