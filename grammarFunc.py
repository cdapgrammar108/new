import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import logging
from keras.preprocessing.sequence import pad_sequences
import requests
from pytorch_pretrained_bert import BertForSequenceClassification
from hunspell import Hunspell
import spacy
import numpy as np
import math
from difflib import SequenceMatcher
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
logging.basicConfig(level=logging.INFO)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def grammerFunc(inputSentence):
    valueOutput = []

    def check_GE(sents):
        """Check of the input sentences have grammatical errors

        :param list: list of sentences
        :return: error, probabilities
        :rtype: (boolean, (float, float))
        """

        # Create sentence) and label lists
        # We need to add special tokens at the beginning and end of each sentence
        # for BERT to work properly
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sents]
        labels = [0]

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # Padding Sentences
        # Set the maximum sequence length. The longest sequence in our training set
        # is 47, but we'll leave room on the end anyway.
        # In the original paper, the authors used a length of 512.
        MAX_LEN = 128

        predictions = []
        true_labels = []

        # Pad our input tokens
        input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
        )

        # Index Numbers and Padding
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # pad sentences
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                                  dtype="long", truncating="post", padding="post")

        # Attention masks
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

        prediction_inputs = torch.LongTensor(input_ids)
        prediction_masks = torch.LongTensor(attention_masks)
        prediction_labels = torch.LongTensor(labels)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = modelGED(prediction_inputs, token_type_ids=None,
                              attention_mask=prediction_masks)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)

        #   print(predictions)
        flat_predictions = [item for sublist in predictions for item in sublist]
        #   print(flat_predictions)
        prob_vals = flat_predictions
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        # flat_true_labels = [item for sublist in true_labels for item in sublist]
        #   print(flat_predictions)
        return flat_predictions, prob_vals

    # load previously trained BERT Grammar Error Detection model

    # from self google drive
    # from google.colab import drive
    # drive.mount('/content/drive')
    # !cp './drive/My Drive/Colab Notebooks/S89A/bert-based-uncased-GED.pth'

    def download_file_from_google_drive(id, destination):
        print("Trying to fetch {}".format(destination))

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in progress_bar(response.iter_content(CHUNK_SIZE)):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)

    def progress_bar(some_iter):
        try:
            from tqdm import tqdm
            return tqdm(some_iter)
        except ModuleNotFoundError:
            return some_iter

    # download_file_from_google_drive("1al7v87aRxebSUCXrN2Sdd0jGUS0zZ3vn", "./bert-based-uncased-GED.pth")

    modelGED = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                             num_labels=2)

    # restore model
    modelGED.load_state_dict(torch.load('S89A'))
    modelGED.eval()

    model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    model.eval()

    tokenizerLarge = BertTokenizer.from_pretrained('bert-large-uncased')

    # !sudo apt-get install libhunspell-1.6-0 libhunspell-dev
    # !pip install cyhunspell

    # download the gn_GB dictionary for hunspell
    download_file_from_google_drive("1jC5BVF9iZ0gmRQNmDcZnhfFdEYv8RNok", "./en_GB-large.dic")
    download_file_from_google_drive("1g8PO8kdw-YmyOY_HxjnJ5FfdJFX4bsPv", "./en_GB-large.aff")

    gb = Hunspell("en_GB-large", hunspell_data_dir=".")

    # List of common determiners
    # det = ["", "the", "a", "an"]
    det = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
           'her', 'its', 'our', 'their', 'all', 'both', 'half', 'either', 'neither',
           'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite']

    # List of common prepositions
    prep = ["about", "at", "by", "for", "from", "in", "of", "on", "to", "with",
            "into", "during", "including", "until", "against", "among",
            "throughout", "despite", "towards", "upon", "concerning"]

    # List of helping verbs
    helping_verbs = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'shall', 'should', 'may', 'might', 'must', 'can', 'could']

    # test sentences

    org_text = []
    org_text.append(inputSentence)

    def create_spelling_set(org_text):
        """ Create a set of sentences which have possible corrected spellings
        """

        sent = org_text
        sent = sent.lower()
        sent = sent.strip().split()

        nlp = spacy.load("en")
        proc_sent = nlp.tokenizer.tokens_from_list(sent)
        nlp.tagger(proc_sent)

        sentences = []

        for tok in proc_sent:
            # check for spelling for alphanumeric
            if tok.text.isalpha() and not gb.spell(tok.text):
                new_sent = sent[:]
                # append new sentences with possible corrections
                for sugg in gb.suggest(tok.text):
                    new_sent[tok.i] = sugg
                    sentences.append(" ".join(new_sent))

        spelling_sentences = sentences

        # retain new sentences which have a
        # minimum chance of correctness using BERT GED
        new_sentences = []

        for sent in spelling_sentences:
            no_error, prob_val = check_GE([sent])
            exps = [np.exp(i) for i in prob_val[0]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            if (softmax[1] > 0.6):
                new_sentences.append(sent)

        # if no corrections, append the original sentence
        if len(spelling_sentences) == 0:
            spelling_sentences.append(" ".join(sent))

        # eliminate dupllicates
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))

        return spelling_sentences

    def create_grammar_set(spelling_sentences):
        """ create a new set of sentences with deleted determiners,
            prepositions & helping verbs

        """

        new_sentences = []

        for text in spelling_sentences:
            sent = text.strip().split()
            for i in range(len(sent)):
                new_sent = sent[:]

                if new_sent[i] not in list(set(det + prep + helping_verbs)):
                    continue

                del new_sent[i]
                text = " ".join(new_sent)

                # retain new sentences which have a
                # minimum chance of correctness using BERT GED
                no_error, prob_val = check_GE([text])
                exps = [np.exp(i) for i in prob_val[0]]
                sum_of_exps = sum(exps)
                softmax = [j / sum_of_exps for j in exps]
                if (softmax[1] > 0.6):
                    new_sentences.append(text)

        # eliminate dupllicates
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))
        return spelling_sentences

    def create_mask_set(spelling_sentences):
        """For each input sentence create 2 sentences
           (1) [MASK] each word
           (2) [MASK] for each space between words
        """
        sentences = []

        for sent in spelling_sentences:
            sent = sent.strip().split()
            for i in range(len(sent)):
                # (1) [MASK] each word
                new_sent = sent[:]
                new_sent[i] = '[MASK]'
                text = " ".join(new_sent)
                new_sent = '[CLS] ' + text + ' [SEP]'
                sentences.append(new_sent)

                # (2) [MASK] for each space between words
                new_sent = sent[:]
                new_sent.insert(i, '[MASK]')
                text = " ".join(new_sent)
                new_sent = '[CLS] ' + text + ' [SEP]'
                sentences.append(new_sent)

        return sentences


    def check_grammar(org_sent, sentences, spelling_sentences):
        """ check grammar for the input sentences
        """

        n = len(sentences)

        # what is the tokenized value of [MASK]. Usually 103
        text = '[MASK]'
        tokenized_text = tokenizerLarge.tokenize(text)
        mask_token = tokenizerLarge.convert_tokens_to_ids(tokenized_text)[0]

        LM_sentences = []
        new_sentences = []
        i = 0  # current sentence number
        l = len(org_sent.strip().split()) * 2  # l is no of sentencees
        mask = False  # flag indicating if we are processing space MASK

        for sent in sentences:
            i += 1

            print(".", end="")
            if i % 50 == 0:
                print("")

            # tokenize the text
            tokenized_text = tokenizerLarge.tokenize(sent)
            indexed_tokens = tokenizerLarge.convert_tokens_to_ids(tokenized_text)

            # Create the segments tensors.
            segments_ids = [0] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Predict all tokens
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)

            # index of the masked token
            mask_index = (tokens_tensor == mask_token).nonzero()[0][1].item()
            # predicted token
            predicted_index = torch.argmax(predictions[0, mask_index]).item()
            predicted_token = tokenizerLarge.convert_ids_to_tokens([predicted_index])[0]

            # second best prediction. Can you used to create more options
            #     second_index = torch.topk(predictions[0, mask_index], 2).indices[1].item()
            #     second_prediction = tokenizer.convert_ids_to_tokens([second_index])[0]

            text = sent.strip().split()
            mask_index = text.index('[MASK]')

            if not mask:
                # case of MASKed words

                mask = True
                text[mask_index] = predicted_token
                try:
                    # retrieve original word
                    org_word = spelling_sentences[i // l].strip().split()[mask_index - 1]
                #         print(">>> " + org_word)
                except:
                    #         print(spelling_sentences[i%l - 1])
                    #         print(tokenized_text)
                    #         print("{0} {1} {2}".format(i, l, mask_index))
                    print("!", end="")
                    continue
                #     print("{0} - {1}".format(org_word, predicted_token))
                # check if the prediction is an inflection of the original word
                #   if org_word.isalpha() and predicted_token not in gb_infl[org_word]:
                #     continue
                # use SequenceMatcher to see if predicted word is similar to original word
                if SequenceMatcher(None, org_word, predicted_token).ratio() < 0.6:
                    if org_word not in list(set(det + prep + helping_verbs)) or predicted_token not in list(
                            set(det + prep + helping_verbs)):
                        continue
                if org_word == predicted_token:
                    continue
            else:
                # case for MASKed spaces

                mask = False
                #     print("{0}".format(predicted_token))
                # only allow determiners / prepositions  / helping verbs in spaces
                if predicted_token in list(set(det + prep + helping_verbs)):
                    text[mask_index] = predicted_token
                else:
                    continue

            #   if org_word == "in":
            #     print(">>>>>> " + predicted_token)
            #   print(tokenized_text)
            #   print(mask_index)

            text.remove('[SEP]')
            text.remove('[CLS]')
            new_sent = " ".join(text)

            #   print(new_sent)
            # retain new sentences which have a
            # minimum chance of correctness using BERT GED
            no_error, prob_val = check_GE([new_sent])
            exps = [np.exp(i) for i in prob_val[0]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            if no_error and softmax[1] > 0.996:
                #     print(org_word)
                #     print(predicted_token)
                #     print(SequenceMatcher(None, org_word, predicted_token).ratio())
                #     print("{0} - {1}, {2}".format(prob_val[0][1], prob_val[0][0], prob_val[0][1] - prob_val[0][0]))

                #     print("{0} - {1:.2f}".format(new_sent, softmax[1]*100) )
                print("*", end="")
                new_sentences.append(new_sent)
        #   print("{0}\t{1}".format(predicted_token, second_prediction))

        print("")

        # remove duplicate suggestions
        spelling_sentences = []
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))

        return spelling_sentences

    # org_text = []
    # with open("./drive/My Drive/Colab Notebooks/S89A/CoNLL_2013_DS.txt") as file:
    #   org_text = file.readlines()

    # predict for each of the test samples

    for sent in org_text:

        print("Input Sentence >>> " + sent)

        sentences = create_spelling_set(sent)
        spelling_sentences = create_grammar_set(sentences)
        sentences = create_mask_set(spelling_sentences)

        print("processing {0} possibilities".format(len(sentences)))

        sentences = check_grammar(sent, sentences, spelling_sentences)

        print("Suggestions & Probabilities")

        if len(sentences) == 0:
            print("None")
            continue

        no_error, prob_val = check_GE(sentences)

        for i in range(len(prob_val)):
            exps = [np.exp(i) for i in prob_val[i]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            print("{0} - {1:0.4f}%".format(sentences[i], softmax[1] * 100))
            valueOutput = sentences[i]

    return valueOutput
