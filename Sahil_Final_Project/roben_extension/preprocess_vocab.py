import argparse
import sys
import itertools
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import wordnet
import nltk
import torch
nltk.download('wordnet')

from utils import pkl_save, pkl_load
from edit_dist_utils import get_all_edit_dist_one, get_all_internal_permutations, get_sorted_word

TOY_VOCAB = ['cat', 'bat', 'car', 'bar', 'airplane!!!']
GLOVE_PATH = 'data/glove/glove.6B.50d.txt'
# Stopwords taken from TextFooler attack
stopwords =  ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

# Used for KNN style clustering based on GloVe distance - very slow and expensive process
'''
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        with open(embedding_name, 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx)
            for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

glove_6b50d = TokenEmbedding(GLOVE_PATH)
'''

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vocab-type', choices=['lm', 'glove'], default = 'glove',
                        help = 'Where to get the vocab from')
  parser.add_argument('--save-root', type = str, default = '',
                        help = 'Name used to format vocab preprocessing output')
  parser.add_argument('--filetype', type = int, default = 1111,
                        help = 'insert, delete, substitute, swap')
  parser.add_argument('--perturb_type', default = 'ed1', type = str, help = 'Type of perturbation to make dict for.')
  return parser.parse_args()

def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = torch.mv(W, x.reshape(
        -1,)) / (torch.sqrt(torch.sum(W * W, dim=1) + 1e-9) * torch.sqrt(
            (x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    results = []
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        results.append(embed.idx_to_token[int(i)])
    return results

def load_glove_vocab(glove_path, num_lines = 400000):
    print("Reading GloVe vectors from {}...".format(glove_path))
    vocab = []
    with open(glove_path) as f:
        for i, line in tqdm(enumerate(f), total=num_lines):
            toks = line.strip().split(' ')
            word = toks[0]
            vocab.append(word)
    return vocab

def vocab_from_lm(lm):
    print("Possible vocab size: ", len(lm.word_to_idx))
    vocab = list(lm.word_to_idx)
    vocab = [word for word in  vocab if word.isalpha() and word == word.lower()]
    print("Vocab size after flitering: ", len(vocab))
    return vocab

def preprocess_neighbors_intprm(vocab):
    neighbor_trans_map = None
    sorted2word = defaultdict(set)
    vocab = [word.lower() for word in vocab]
    print("Grouping by sorted word")
    for word in tqdm(vocab):
        sorted_word = get_sorted_word(word)
        sorted2word[sorted_word].add(word)

    neighbor_trans_map = None
    print("Constructing edges...")
    neighbor_map = defaultdict(set)
    for sorted_word in tqdm(sorted2word):
        permutations = itertools.permutations(sorted2word[sorted_word], r = 2)
        for src, dest in permutations:
            neighbor_map[src].add(dest)
        #Allow self-edges
        for src in sorted2word[sorted_word]:
            neighbor_map[src].add(src)
    return sorted2word, neighbor_map, neighbor_trans_map



def preprocess_neighbors(vocab, filetype = 1111, sub_restrict = None):
    #For efficiency, assume edit distance 1 is symmetric. Not true for certain filetypes, so perturbations act accordingly...
    typo2vocab = defaultdict(set)
    print("Making typo dict...")
    for word in tqdm(vocab):
        perturbations = get_all_edit_dist_one(word, filetype = filetype, sub_restrict = sub_restrict)
        # Add synonyms Wordnet - all synonyms for each word
        for syn in wordnet.synsets(word):
            if word in stopwords:
                continue
            i = 0
            for l in syn.lemmas():
                if l.name() not in stopwords and i < 3:
                    typo2vocab[l.name()].add(word)
                i = i+1
        # Add synonyms GloVe embeddings 
        # synonyms = get_similar_tokens(word, 3, glove_6b50d)
        # for syn in synonyms:
        #    typo2vocab[syn].add(word)
        # for typo in perturbations:
        #    typo2vocab[typo].add(word)

    print("Constructing edges...")
    neighbor_map = defaultdict(set)
    neighbor_trans_map = defaultdict(set)
    for typo in tqdm(typo2vocab):
        permutations = itertools.permutations(typo2vocab[typo], r = 2)
        for src, dest in permutations:
            neighbor_map[src].add(dest)
            neighbor_trans_map[(src, dest)].add(typo)
        #Allow self-edges
        for src in typo2vocab[typo]:
            neighbor_map[src].add(src)
            neighbor_trans_map[(src, src)].add(src)
    return typo2vocab, neighbor_map, neighbor_trans_map



def preprocess_vocab(args):
    if args.vocab_type == 'glove':
        vocab = load_glove_vocab(GLOVE_PATH)
    elif args.vocab_type == 'lm':
        query_handler = load_language_model()
        vocab = vocab_from_lm(query_handler)
    else:
        raise ValueError("Invalid vocab type of {}".format(args.vocab_type))
    print("Vocab sample: ", vocab[300:500])
    sub_dict = None
    if args.modify_end:
        print("Modifying the end...")
    if args.perturb_type == 'ed1':
        typo2vocab, ed2_neighbors, neighbor_trans_map = preprocess_neighbors(vocab, filetype = args.filetype, 
                                                                        sub_restrict = sub_dict)
    elif args.perturb_type == 'intprm':
        typo2vocab, ed2_neighbors, neighbor_trans_map = preprocess_neighbors_intprm(vocab)
    else:
        raise NotImplementedError
    #print("Typo2vocab bos: ", typo2vocab['bos'])
    print(len(typo2vocab))
    print(len(ed2_neighbors))
    #pkl_save(typo2vocab, 'typo2vocab.pkl')
    pkl_save(ed2_neighbors, 'ed2_neighbors{}pt{}.pkl'.format(args.save_root, args.perturb_type))
    print("Saved ed2")
    #pkl_save(neighbor_trans_map, 'neighbor_trans_map{}.pkl'.format(args.save_root))
    #print("Saved neighbor trans map")

def get_neighbors(args):
    neighbor_path = 'ed2_neighbors{}pt{}.pkl'.format(args.save_root, args.perturb_type)
    neighbor_dict = pkl_load(neighbor_path)
    while True:
        print('broad' in neighbor_dict['bold'])
        inpt = input("Enter a word: ")
        inpt = inpt.lower()
        if inpt not in neighbor_dict:
            print("Word not preprocessed...")
        else:
            print("Neighbors for {}:".format(inpt))
            print(neighbor_dict[inpt])

if __name__ == '__main__':
    args = parse_args()
    #preprocess_vocab(args)
    get_neighbors(args)
    #preprocess_vocab(args)
    #preprocess_vocab(args.vocab_type, save_root = '_test')
    #a = generate_qwerty_dict()
    #for elem in a:
    #   print("{}: {}".format(elem, a[elem]))
