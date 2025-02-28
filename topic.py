import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import nltk
import re
import spacy
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic import BERTopic
import os
from llamaguard3 import moderate 


# Preprocess text (Tokenize, Remove Stopwords & Punctuation)
stop_words = set(stopwords.words("english"))
def preprocess_text(text):
    # remove urls
    #text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # remove numbers
    #text = re.sub(r"\d+", "", text) 
    # remoe punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # convert all to lowercase and tokenize
    tokens = word_tokenize(text.lower())  
    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # perform lemmatization-groups same term with different forms together
    tokens = [token.lemma_ for token in nlp(" ".join(tokens)) if token.lemma_ not in stop_words]
    return tokens


def ldaWholeConvo(df):
    
    # Convert Each Row (1 conversation rollout) into a single document (1 string)
    # this joins all convo turns into one doc
    convos = df.apply(lambda row: " ".join(row.astype(str)), axis=1) 

    # preprocess text into cleaned tokens
    tokenized_convos = convos.apply(preprocess_text)

    # create dict and corpus (Bag-of-Words rep)
    dictionary = corpora.Dictionary(tokenized_convos)  # Create dictionary
    corpus = [dictionary.doc2bow(convo) for convo in tokenized_convos]  # Convert docs to BoW

    # train imported LDA model on our conversations
    num_topics = 25 
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha="auto", eta='auto')

    topics_list = []
    # display and save topics lda model found
    for idx, topic in lda_model.print_topics():
        # create dataframe with the 15 topics and what terms make up each topic
        topics_list.append({"Topic_Number": idx, "Top_Terms": topic})
        print(f"Topic {idx}: {topic}")

    foundTopics_df = pd.DataFrame(topics_list)

    # assign topics to each coversation
    topic_assignments = [lda_model[convo] for convo in corpus]

    # convert topic distributions to a DataFrame
    topic_df = pd.DataFrame([{f"Topic_{t[0]}": t[1] for t in convo} for convo in topic_assignments])

    # concatenate topic results with original conversations
    result_df = pd.concat([df, topic_df], axis=1)

    # Save results to csv
    result_df.to_csv(os.path.join(output_folder,output_file+"_lda_full_w_topics.csv"), index=False)
    #topic_df.to_csv("BigNewMethod_topicDistrib.csv", index = False)
    foundTopics_df.to_csv(os.path.join(output_folder,output_file+"_lda_full_topics.csv"), index=False)

def ldaPerCol(df):
    # Initialize storage for topic distributions
    topic_distributions = {}
    topicsPerCol = {}

    # Process each column individually
    num_topics = 15  # Define number of topics
    top_n_topics = 10  # Define number of top topics to extract

    for column in df.columns:
        texts = df[column].dropna().astype(str).tolist()
        tokenized_texts = [preprocess_text(text) for text in texts]

        # Skip columns with no meaningful text after preprocessing
        if not any(tokenized_texts):
            print(f"Skipping column '{column}' - No valid text after preprocessing.")
            continue
        
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        # Ensure corpus is not empty before training LDA
        if len(corpus) == 0 or all(len(doc) == 0 for doc in corpus):
            print(f"Skipping column '{column}' - No valid terms in corpus.")
            continue
        
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha="auto", eta='auto')
        
        # Extract topic distribution
        topics = lda_model.show_topics(num_topics=top_n_topics, formatted=False)
        topicsPerCol[column] = topics
        topic_distributions[column] = {f"Topic_{i}": " ".join([word[0] for word in topic[1]]) for i, topic in enumerate(topics)}

    # Convert topic distributions to a DataFrame
    if topic_distributions:
        topic_df = pd.DataFrame(topic_distributions)
        topic_df.to_csv(os.path.join(output_folder,output_file+"_lda_w_topics_per_column.csv"), index=False)
    else:
        print("No valid topics were extracted. Check the input data.")

    if topicsPerCol:
        topicsPerCol_df = pd.DataFrame(topicsPerCol)
        topicsPerCol_df.to_csv(os.path.join(output_folder,output_file+"_lda_topics_per_column*.csv"), index=False)

def bertTopicWholeConvo(df):
    # code for this function referenced heavily from: https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing#scrollTo=Fo-Oig4Yib5K
    # Convert each row (conversation rollout) into a single string
    convos = df.apply(lambda row: " ".join(row.astype(str)), axis=1) 

    # Split conversations into sentences for better topic modeling
    sentences = [sent_tokenize(convo) for convo in convos]
    sentences = [sentence for convo in sentences for sentence in convo]  # Flatten list

    # Pre-calculate Embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(sentences, show_progress_bar=True)
    
    # Reduce dimensionality of embeddings and prevent stochastic behavior
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # Create clustering model and control number of topics (150 -30)
    hdbscan_model = HDBSCAN(min_cluster_size=70, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # Preprocess topic representations
    vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

    # Multi-aspect topic modeling
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech("en_core_web_sm")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # Representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model,
        "POS": pos_model
    }
    
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True
    )
    
    # Train BERTopic on sentence-level data
    topics, probs = topic_model.fit_transform(sentences, embeddings)
    
    # Get topic information as a DataFrame
    topic_info_df = topic_model.get_topic_info()
    topic_info_df.to_csv(os.path.join(output_folder,output_file+"_BERTTopic_full_Topics.csv"), index=False)

    # Assign topics to original conversations
    df["Assigned_Topic"] = [topics[i] for i in range(len(convos))]  # Map sentence topics back to convos
    df["Topic_Probability"] = [probs[i] for i in range(len(convos))]

    # Save conversation assignments
    df.to_csv(os.path.join(output_folder,output_file+"_BERTTopic_full_AssignedTopics.csv"), index=False)

def bertTopicPerCol(df):
    # code for this function referenced heavily from: https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2?usp=sharing#scrollTo=Fo-Oig4Yib5K
    # Initialize storage for results
    all_topics_list = []

    # Process each column/turn in conversation
    for column in df.columns:
        # Drop NaN values and convert to string
        texts = df[column].dropna().astype(str).tolist()

        # Skip columns with too few valid texts
        if len(texts) < 10:
            print(f"Skipping column '{column}' - Not enough text data.")
            continue

        # Join conversation turns per row into one string
        convos = [" ".join(text.split()) for text in texts]

        # Split conversations into sentences 
        sentences = [sent_tokenize(convo) for convo in convos]
        sentences = [sentence for convo in sentences for sentence in convo]  # Flatten list
        
        # Skip if not enough sentences
        if len(sentences) < 10:
            print(f"Skipping column '{column}' - Not enough sentences after tokenization.")
            continue

        # Pre-calculate Embeddings
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(sentences, show_progress_bar=True)

        # Reduce dimensionality of embeddings
        umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.1, metric='cosine', random_state=42)

        # Control number of topics with HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)

        # Improve topic representation
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))

        # Multi-aspect topic modeling
        keybert_model = KeyBERTInspired()
        pos_model = PartOfSpeech("en_core_web_sm")
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # Define topic representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
            "POS": pos_model
        }

        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            verbose=True
        )

        # Train BERTopic on sentence-level data
        topics, probs = topic_model.fit_transform(sentences, embeddings)

        # Get topic information and add column name
        topic_info_df = topic_model.get_topic_info()
        topic_info_df["Column"] = column  # Add column name for tracking

        # Append topics to list for later merging
        all_topics_list.append(topic_info_df)

        # # Save topic information for this column
        # topic_info_filename = f"BERTopic_TopicInfo_{column}.csv"
        # topic_info_df.to_csv(topic_info_filename, index=False)

        # Assign topics to original conversations
        df[f"Assigned_Topic_{column}"] = [topics[i] if i < len(topics) else -1 for i in range(len(df))]
        df[f"Topic_Probability_{column}"] = [probs[i] if i < len(probs) else -1 for i in range(len(df))]


    # Save full DataFrame with topic assignments
    #df.to_csv(os.path.join(output_folder,"New5t_BERTopic_AssignedTopics_PerColumn2.csv"), index=False)
    if all_topics_list:
        merged_topics_df = pd.concat(all_topics_list, ignore_index=True)
        merged_topics_df.to_csv(os.path.join(output_folder,output_file+"_Bertopic_topics*.csv"), index=False)

import pandas as pd

def evalTrajectories(df):
    # Create an empty dictionary to store results
    results_dict = {}

    # Loop through every row and column
    for index, row in df.iterrows():
        row_results = {}
        for col in df.columns:
            try:
                # Ensure the function only runs on valid strings
                if isinstance(row[col], str):
                    _, labelList = moderate(row[col])
                else:
                    labelList = None  # Assign None if not a string
            except Exception as e:
                print(f"Error processing row {index}, column '{col}': {e}")
                labelList = None  # Assign None in case of an error
            
            row_results[col] = labelList
        
        # Store row results in dictionary
        results_dict[index] = row_results

    # Convert dictionary to DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient="index")

    # Save processed results
    results_df.to_csv(os.path.join(output_folder,output_file"_LlamaGuard_TrajEvals*.csv"), index=False)

    #return results_df  # Return DataFrame if needed


if __name__ == "__main__":
    # Make sure to set...
        #  correct output folder name
        #  correct outptt file name (will work for all functions as save results to csv) 
        # correct dataframe to read in
    
    # set folder to output csv files to
    output_folder = "Llama3TNewMethod"
    output_file = "NewLlama"
    os.makedirs(output_folder, exist_ok=True)
    # download NLTK stopwords and punctuation
    nltk.download("stopwords")
    nltk.download("punkt")
    # import pre-processed nlp model trained for pre-processing text
    nlp = spacy.load("en_core_web_lg")

    # read in csv of conversation rollouts as a pandas dataframe
    #df = pd.read_csv("/home/allie11/ASTPrompter/DataFrames/Baseline/BigBaselineNewR.csv") 
    #df = pd.read_csv("/home/allie11/ASTPrompter/5TBigBaseline.csv")
    #df = pd.read_csv("/home/allie11/ASTPrompter/Dataframe New Method/BigNewMethod.csv") 
    df = pd.read_csv("/home/allie11/ASTPrompter/NewLlamaTs.csv") 

    # call topic modeling and llamaguard eval functions
    bertTopicPerCol(df)
    bertTopicWholeConvo(df)
    ldaPerCol(df)
    ldaWholeConvo(df)
    evalTrajectories(df) # per column
    print("Done")           