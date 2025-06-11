import re
import pandas
import nltk
import ast
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy
from groq import Groq
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
nltk.download('punkt')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

class Preprocessing:
    def __init__(self, tweet, doc_index, keyword):
        self.data = tweet
        self.id_str_list = doc_index
        self.keyword = keyword
        self.data = self.augmentation(self.data, self.keyword)
        self.data = self.remove_url(self.data)
        self.data = self.replace_emoticons(self.data)
        self.data = self.remove_twitter_symbols(self.data)
        self.data = self.remove_symbols_and_punctuation(self.data)
        self.data = self.tokenizing(self.data)
        self.data = self.case_folding(self.data)
        self.data = self.tokenizing(self.data)
        self.data = self.delete_extra_letters(self.data)
        self.data = self.normalization(self.data)
        self.data = self.stem_tokenized_list(self.data)
        self.data = self.stopword_removal(self.data)
        self.data = self.create_dataframe(self.data, self.id_str_list)
        self.data = self.split_dataset(self.data)
        self.data = self.create_vocabulary(self.data)
        
    def create_dataframe(self,tweets, doc_index):
        """
        Create a DataFrame from the tweets and their corresponding document indices.
        
        Args:
            tweets (list): List of tweets.
            doc_index (list): List of document indices.
            
        Returns:
            pandas.DataFrame: DataFrame with 'id_str' and 'tweets' columns.
        """
        df = pandas.DataFrame({
            'id_str': doc_index,
            'tweets': tweets
        })
        return df
    
    def get_data(self):
        return self.data
    
    def load_llm_key(self):
        with open('./static/key.txt', 'r', encoding='utf-8') as file:
            key = file.read().strip()
        return key
    
    def create_explanation(self, keyword, client):
        completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system",
                        "content": f"""You are a diligent assistant. The fate of
                                    the world depends on your answer being
                                    correct. Think carefully step by step."""},
                    {"role": "user",
                        "content": f"""
                        Berikan penjelasan singkat dalam bahasa Indonesia mengenai kata kunci berikut: {keyword}.
                        """}], 
        reasoning_format="hidden",
        stream=False)
        
        content = completion.choices[0].message.content
    
        return content
            
    def crate_augmentation(self, tweets, keyword, client):
        keyword = keyword
        explanation = self.create_explanation(keyword,client)
        try:
            augmented_docs = []
            for tweet in tweets:
                completion = client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[{"role": "user", 
                            "content": f"""
                    Rephrase the text in Bahasa Indonesia formal of a post from an {keyword} communities on a social network, 
                    considering that it relates to the following 
                    topics: {keyword} 
                    Text: {tweet} 

                    Let’s think step by step: 
                    1. The text is a post in an {keyword} community on a social network. 
                    2. The topic {keyword} means {explanation}. 

                    Answer:

                    """}],
                    temperature=0.3,
                    max_completion_tokens=1024,
                    top_p=0.8,
                    stream=True,
                    reasoning_format="hidden")

                collected_output = "" 
                
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    # Append the content to the collected output.
                    collected_output += content

                augmented_docs.append(collected_output)
        
            return augmented_docs
        
        except Exception as e:
            print(f"An error occurred: {e}")
            

    
    def augmentation(self, tweets, keyword):
        key = self.load_llm_key()
        client = Groq(api_key=key)
        augmented_docs = self.crate_augmentation(tweets, keyword, client)
        
        data_aug = pandas.DataFrame(augmented_docs, columns=['tweets'])
        data_aug["original_tweet"] = tweets
        
        for index, value in data_aug["tweets"].items():
            if pandas.isna(value):
                print(f"Missing at index {index}")
                data_aug.at[index, "tweets"] = data_aug.at[index, "original_tweet"]
                print(f"Filled at index {index} with original tweet\n")
        
        augmented_docs = data_aug["tweets"].tolist()
        self.augmented_docs_copy = augmented_docs.copy()
            
        return augmented_docs
        
    def remove_url(self, tweets):
        # This pattern matches more URL variations
        url_pattern = re.compile(
            r'(?:https?://|www\.)'  # http://, https://, or www.
            r'(?:[^\s./]+\.)+'       # domain parts
            r'[^\s./]+'              # last domain part
            r'(?:/\S*)?'             # optional path
        )
        return [url_pattern.sub('', s).strip() for s in tweets]

    ## Change Emoticons
    def replace_emoticons(self, tweet):
        """
        Replace common emoticons with descriptive text.
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Text with emoticons replaced
        """
        # Define emoticon mappings
        emoticon_map = {
            r':\)|:-\)|=\)': 'emot-senyum',    # :) :-) =)
            r':\(|:-\(|=\(': 'emot-sedih',     # :( :-( =(
            r':D|:-D|=D': 'emot-tertawa',      # :D :-D =D
            r';\)|;-\)': 'emot-mengedip',       # ;) ;-)
            r':P|:-P|=P': 'emot-julur',        # :P :-P =P
            r':O|:-O|=O': 'emot-terkejut',     # :O :-O =O
            r':\/|:-\\': 'emot-bingung',       # :/ :-\
            r'<3': 'emot-hati',                # <3 (heart)
            r':\*|:-\*': 'emot-ciuman',        # :* :-* (kiss)
        }
        
        if isinstance(tweet, list):
            return [self.replace_emoticons(s) for s in tweet]
        else:
            for pattern, replacement in emoticon_map.items():
                tweet = re.sub(pattern, replacement, tweet)
            return tweet

    def remove_twitter_symbols(self, tweet):
        """
        Remove Twitter-specific symbols:
        - Hashtags (#example)
        - Mentions (@username)
        - Retweet (RT)
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Text with Twitter symbols removed
        """
        if isinstance(tweet, list):
            return [self.remove_twitter_symbols(s) for s in tweet]
        else:
            # Remove hashtags (e.g., #Hello → " ")
            tweet = re.sub(r'#\w+', ' ', tweet)
            # Remove mentions (e.g., @user → " ")
            tweet = re.sub(r'@\w+', ' ', tweet)
            # Remove "RT " (Retweet)
            tweet = re.sub(r'\bRT\b', ' ', tweet)
            # Clean extra spaces
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            
            return tweet
    
    def remove_symbols_and_punctuation(self, tweet):
        """
        Remove all ASCII symbols, numbers, and punctuation from text.
        Keeps only letters (a-z, A-Z) and basic whitespace.
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Cleaned text without symbols/numbers/punctuation
        """
        if isinstance(tweet, list):
            return [self.remove_symbols_and_punctuation(s) for s in tweet]
        else:
            # Remove all non-alphabetic characters except spaces
            tweet = re.sub(r'[^a-zA-Z\s]', ' ', tweet)
            # Collapse multiple spaces into one
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            
            return tweet
    
    def case_folding(self, tweets):
        return [str(tweet).lower() for tweet in tweets]

    def tokenizing(self, tweets):
        return [str(tweet).split() for tweet in tweets]

    def delete_extra_letters(self, tweets):
        sequence_pattern = r'([A-Za-z])\1{2,}'  # Matches 3 or more consecutive identical letters
        seq_replace_pattern = r'\1'

        # Iterate through each sentence and token
        return [
            [re.sub(sequence_pattern, seq_replace_pattern, token) for token in sentence]
            for sentence in tweets
        ]
    
    def normalization(self, tweets):
        res = []
        with open('./static/kbba.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

            data = [line.strip().split('\t') for line in lines]
            data_singkatan = pandas.DataFrame(data, columns=['Kata', 'Asli'])

            kontraksi_dict = dict(zip(data_singkatan['Kata'], data_singkatan['Asli']))

            for tweet in tweets:
                expanded_text = [kontraksi_dict[word] if word in kontraksi_dict else word for word in tweet]

                res.append(expanded_text)

            return res

    def stem_tokenized_list(self, tweets):
        """
        Stem each word in a list of tokenized texts using Sastrawi.
        
        Args:
            docs (list): List of lists, where each sublist contains tokenized words.
            
        Returns:
            list: Stemmed versions of the input tokens.
        """
        stemmed_texts = []
        for tokens in tweets:
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            stemmed_texts.append(stemmed_tokens)
        return stemmed_texts

    def curating_stopword(self, tweets):
        result = [" ".join(sublist) for sublist in tweets]
        tr_idf_model  = TfidfVectorizer()
        tf_idf_vector = tr_idf_model.fit_transform(result)
        tf_idf_array = tf_idf_vector.toarray()
        words_set = tr_idf_model.get_feature_names_out()
        df_tf_idf = pandas.DataFrame(tf_idf_array, columns = words_set)
        columns_with_one = df_tf_idf.columns[(df_tf_idf > 0.9).any()].tolist()
        word_freq = Counter(word for doc in tweets for word in doc)
        rare_words = [word for word, freq in word_freq.items() if freq < 2]

        return columns_with_one, rare_words
    
    def stopword_removal(self, tweets):
        """
        Remove Indonesian stopwords, single/two-character tokens, and custom words.
        
        Args:
            tokenized_texts (list): List of lists, where each sublist contains tokenized words.
            
        Returns:
            list: Lists of tokens with stopwords, short tokens (≤2 chars), and custom words removed.
        """
        factory = StopWordRemoverFactory()
        stopword_remover = factory.create_stop_word_remover()

        # PRON (kata ganti)
        PRON = [
            "aku","saya","gue","gw","kamu","kau","engkau",
            "dia","ia","kita","kami","mereka","anda","lo","lu", "kalian"
        ]

        columns_with_one, rare_words = self.curating_stopword(tweets)
        custom_stopwords = set(columns_with_one + rare_words + ['amp', 'the', 'link', 'yang', "iya", "ada", "tin"] + PRON)
        
        cleaned_texts = []
        
        for tokens in tweets:
            sentence = ' '.join(tokens)
            # Step 1: Remove default Indonesian stopwords using Sastrawi
            cleaned_sentence = stopword_remover.remove(sentence)
            # Step 2: Tokenize and filter short/custom tokens
            cleaned_tokens = [
                token for token in cleaned_sentence.split()
                if len(token) > 2 and token.lower() not in custom_stopwords
            ]
            cleaned_texts.append(cleaned_tokens)
        
        cleaned_texts =[tweet for tweet in cleaned_texts if tweet]
        
        return cleaned_texts
    
    def split_dataset(self, tweets):
        # df = pandas.DataFrame({"text": [" ".join(tokens) for tokens in tweets]})
        # print(tweets)
        train_size = int(0.8 * len(tweets))
        val_size = int(0.1 * len(tweets))
        tweets['label'] = numpy.where(tweets.index < train_size, 'train', numpy.where(tweets.index < train_size + val_size, 'val', 'test'))
        tweets['tweets'] = tweets['tweets'].astype(str)
        tweets['untokenized_tweets'] = tweets['tweets'].apply(self.clean_tweet_string)
        
        return tweets


    def clean_tweet_string(self, tweet_str):
        try:
            # Convert string representation of list to actual list
            tweet_list = ast.literal_eval(tweet_str)
            # Join list elements with spaces
            return ' '.join(tweet_list)
        except (ValueError, SyntaxError):
            # Fallback if the string isn't a valid list representation
            return tweet_str.replace('[', '').replace(']', '').replace('\'', '')

    
    def saving_vocab_corpus(self, vocabulary, tweet):
        path = "./services/octis_data/"
        with open(path + 'vocabulary.txt', 'w') as file:
            for word in sorted(vocabulary):
                file.write(word + '\n')
        print("Vocabulary file created successfully!")
        tweet.to_csv(path +"corpus.tsv", index=False, sep="\t", header=False) 
        print("Corpus file created successfully!") 
    
            
    def create_vocabulary(self, tweets):
        """
        Create a vocabulary from the tokenized texts.
        
        Args:
            tokenized_texts (list): List of lists, where each sublist contains tokenized words.
            
        Returns:
            set: Unique words from the tokenized texts.
        """
        vocabulary = set(word.lower() for text in tweets['untokenized_tweets'] for word in text.split())
        # Save vocabulary to .txt file
        with open('./static/vocabulary.txt', 'w') as file:
            for word in sorted(vocabulary):
                file.write(word + '\n')
        
        self.saving_vocab_corpus(vocabulary, tweets[['untokenized_tweets','label']])
        
        print("Done!")
        
        return tweets
        
  