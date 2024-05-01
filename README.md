# FineGrainedLLMDetox

Applying [FineGrained RLHF](https://arxiv.org/pdf/2306.01693) to dialogue

## Set up instructions
1. Create a new conda environment
`conda create --name detox`
2. Activate the conda environment
`conda activate detox`
3. Install the requirements
`pip3 install -r requirements.txt`
4. Set up .env file with Perspective credential
`echo PERSPECTIVE_API_KEY="insert your key here" > .env` 