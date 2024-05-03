"""
main.py
core inference routines
"""

# elicitation options
# use anon0 + anon1 + reddit database. give a few turns from the database, then generate
# pros: no risk of inserting toxicity that is abrupt / doesn't make sense in context
# cons: may be slow/rare to get to toxicity, construct validity (is anon0/anon1 okay?)
# 2. anon0 + anon1 + reddit, then insert realToxicity prompt
# pros: will definitely get toxic
# cons: prompt may not fit context; potential for model to learn something that doesn't generalize as well b/c of the set up
# 3. anon0 + anon1 + reddit + state a persona for each user (following the very popular persona chat paper, except we could give personas which are more likely to lead to undesirable behavior - this is justified by saying that people like this commonly engage online)
# pros: avoids the concerns of #2
# cons: potentially lacks construct validity, not clear if this will get toxic (or for which personas)

from lm import LanguageModel
from convokit import Corpus, download
from reward import reward

# download the conversation corpus
corpus = Corpus(filename=download("reddit-corpus-small"))
# initialize the LM
model = LanguageModel()

# sample a random starting conversation
convo = corpus.random_conversation()
convo = [i.text for i in convo.iter_utterances()][1:] # because 0 is the selfpost
# stick together into two turn conversations with anon0:, anon1: etc.
convo = [f"anon{int(i % 2 == 0)}: {text.strip()}" for i, text in enumerate(convo)]
# get the last n lines
length = len(convo)
convo = convo[-5:]

# create a forward call for the trajectory
def get_multiple_uts(convo_prompt, model, n=5):
    # shallow copy convo
    convo = convo_prompt[:]

    for _ in range(n):
        # get the number of conversations remaining
        length = len(convo)
        # create prompts
        prompt = "\n".join(convo+[f"anon{int(length % 2 == 0)}: "]).strip()
        ut = model(prompt)
        convo.append(f"anon{int(length % 2 == 0)}: {ut.strip()}")

    return convo[len(convo_prompt):]

print("\n".join(convo))
print("\n".join(get_multiple_uts(convo, model)))

print(ut)

