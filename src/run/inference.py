from tqdm import tqdm
import json
import os
import torch
import logging
from typing import List

from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.collections import nlp as nemo_nlp

# set up logging in a nice readable format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class Inference:
    '''
        a class to do inference with the pretrained nemo punctuation model, logs the queries (text without punctuation) and the inference (text with predicted punctuation position)
    '''

    def __init__(self, punctuation_model_dir: str, query_list: List[str]) -> None:
        '''
            punctuation_model_dir: the directory where the punctuation model is stored
            query_list: the list of queries for punctuation inference  
        '''

        self.query_list = query_list
        self.pretrained_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(punctuation_model_dir)

    def infer(self) -> None:
        '''
            performs the inference        
        '''

        inference_list = self.pretrained_model.add_punctuation_capitalization(self.query_list)

        for query, result in tqdm(zip(self.query_list, inference_list)):
            logging.getLogger('QUERY').info(query)
            logging.getLogger('RESULT').info(f'{result}\n')

    def __call__(self):
        return self.infer()

if __name__ == '__main__':
    
    i = Inference(
        punctuation_model_dir='punctuation_en_distilbert',
        query_list=[
            'we bought four shirts and one mug from the nvidia gear store in santa clara',
            'what can i do for you today',
            'how are you',
            'how is the weather in',
            'how is rachel doing is she fine'
        ]
    )

    i()