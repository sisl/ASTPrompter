from dotenv import load_dotenv
from googleapiclient import discovery
import json
import os
import time

load_dotenv()
API_KEY = os.environ.get("PERSPECTIVE_API_KEY")

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

"""
Process perspective response

input
--------------------------------------------------------------
text (str):                  text input to Perspective API
perspective_response (Dict): response from Perspective API

output
--------------------------------------------------------------
text_to_score (Dict[Dict]): mapping from text (str) to a 
Dict of (attribute, score) pairs
"""
def process_response(text, perspective_response):
    text_to_score = {}
    attribute_scores = perspective_response["attributeScores"]

    for attribute in attribute_scores.keys():
        attribute_score = attribute_scores[attribute]
        span_scores = attribute_score["spanScores"]
        
        for span in span_scores:
            # Map span to text
            begin, end = span["begin"], span["end"]
            span_text = text[begin:end]
            score = span["score"]["value"]
            
            if span_text in text_to_score.keys():
                text_to_score[span_text][attribute] = score
            else:
                text_to_score[span_text] = {attribute: score}
        
        # Add summaryScore for full text
        if text in text_to_score.keys():
            text_to_score[text][attribute] = attribute_score["summaryScore"]["value"]
        else:
            text_to_score[text] = {attribute: attribute_score["summaryScore"]["value"]}

    # print(text_to_score)
    return text_to_score

"""
Call Perspective API

inputs
---------------------------------------------------------------
text (str):            the text to analyze

spanAnnotations (bool): whether to return annotations at a span 
(sentence) level

attributes (Dict):      Dict mapping attribute to reponse 
                        configuration. Possible attributes are 
                        listed below. For details see 
                        https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages?language=en_US
                        - TOXICITY
                        - SEVERE_TOXICITY
                        - IDENTITY_ATTACK
                        - INSULT
                        - PROFANITY
                        - THREAT

outputs
----------------------------------------------------------------
Dict[Dict] mapping text -> {attribute: score} for each attribute
in attributes. if spanAnnotations is True, then will have a key
for each sentence in the text, as well as for the entire text.
if False, then will only report scores for the full text.
"""
def call_perspective(text, span_annotations=False, attributes={"TOXICITY": {}}):
    # input text, call perspective API, do post-processing to get the most useful attributes etc. 
    request = {
        'comment': {'text': text},
        'spanAnnotations': span_annotations,
        'requestedAttributes': attributes
    }
    # to ensure API happiness
    time.sleep(5)

    response = client.comments().analyze(body=request).execute()
    return process_response(text, response)

# Your tests here!
# call_perspective("hello, my name is Amelia. I go to Stanford.", True, {"TOXICITY": {}})
