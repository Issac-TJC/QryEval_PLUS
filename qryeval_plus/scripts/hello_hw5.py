"""
A simple example to illustrate the use of FAISS for ranking documents
and T5 for generating answers.
"""

# Copyright (c) 2025, Carnegie Mellon University.  All Rights Reserved.

import faiss
import json
import numpy
import re
import socket
import torch

from transformers import AutoTokenizer, AutoModel

from qryeval_plus.core.Idx import Idx


# ------------------ Global variables ---------------------- #

model_max_sequence_length = 512				# Max WordPiece tokens
trecEvalOutputLength = 10

index_path = "INPUT_DIR/index-cw22b-wp"
dense_index_path = "INPUT_DIR/index-cw22b-wp-faiss-b300-Fp"
dpr_model_path = "INPUT_DIR/co-condenser-marco-retriever/"
llm_address = "128.2.204.71/59596"


# ------------------ Methods ------------------------------- #

def dense_encode(input_dict):
    """
    Encode a token sequence.
    Input: the tokenized sequence.
    Output: the sense representation.
    """
    with torch.no_grad():
        outputs = dense_model(**input_dict)
        rep = outputs.last_hidden_state[:,0]	# The hidden state of [CLS]
        rep = rep.squeeze()			# [1, 768] -> [768]
        rep = rep.tolist()			# Avoid tensor memory leak
    return(rep)


def dense_tokenize_string(s):
    """
    Use the model to tokenize s, convert to token ids, and return as tensors.
    Input: a text string.
    Output: a dictionary of tensors that BERT understands.
        "input_ids": The ids for each token. 
        "token_type_ids": The token type (sequence) id of each token.
        "attention_mask": For each token, mask(0) or don't mask(1). Not used.
    """
    return(dense_tokenizer.encode_plus(
        s,				# sequence
        max_length=model_max_sequence_length,
        truncation=True,		# Truncate if too long
        return_tensors="pt"))		# Return PyTorch tensors


def send_to_llm(llm_address, messages):
    """
    Send a list of messages to an LLM server and get a response.

    llm_address: IP address / port string, e.g., 128.2.204.71/59596
    messages: A dict of messages that prompts an LLM to perform a task.

    Returns a string with the LLM response or an error message.
    (a string with the prefix 'EXCEPTION:: ').
    """

    max_msg_bytes = 20*1024		# Also enforced by the server.
        
    try:
        # Messages are encoded json terminated by 4 null bytes
        message = json.dumps(messages)
        message = message.encode() + bytearray(b'\0\0\0\0')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            llm_host, llm_port = llm_address.split('/')
            sock.connect((llm_host, int(llm_port)))
            sock.sendall(message)

            # Receive a message terminated by 4 null bytes. It may take
            # multiple calls to recv to get the complete message.
            data = bytearray()
            while len(data) < max_msg_bytes:
                packet = sock.recv(max_msg_bytes - len(data))
                if not packet:		# Socket closed unexpectedly
                    break
                data.extend(packet)
                if data[-4:] == bytearray(b'\0\0\0\0'):
                    data = data[0:-4]	# Discard the 4 null bytes
                    break

            response = data.decode()	# Reconstruct the response
    except Exception as e:
        response = f'EXCEPTION:: {str(e)}'

    # Post-process the LLM server response. You may want to add other
    # post-processing here.
    response = response.strip()			# Remove unnecessary whitespace
    response = re.sub(r'\s+', ' ', response)	# Remove unnecessary whitespace
    response = re.sub(r'\\', '\\\\', response)	# Escape \
    response = re.sub(r'\"', '\\"', response)	# Escape "

    return(response)


def text_truncate(s):
    """Truncate a string to model_max_sequence_length tokens."""
    return(" ".join(s.split()[:model_max_sequence_length]))


# ------------------ Script body --------------------------- #

question = "Do cigarettes cause cancer?"

print('==> Retrieval <==', flush=True)
print(f'Query: {question}', flush=True)

# Initialize retrieval
Idx.open(index_path)
faiss_index = faiss.read_index(dense_index_path)

dense_tokenizer = AutoTokenizer.from_pretrained(dpr_model_path)
dense_model = AutoModel.from_pretrained(dpr_model_path)
dense_model.eval()

# Tokenizing and encoding a string is similar to HW4
encoded_query = dense_encode(dense_tokenize_string(question))

# FAISS evaluates a list of queries. Our list has just 1 query.
encoded_query = [encoded_query]
scores, docids = faiss_index.search(numpy.array(encoded_query),
                                    trecEvalOutputLength)

print(f'Internal docids: {docids[0]}', flush=True) 
print(f'Scores: {scores[0]}', flush=True)

# Get (a very simple) first passage from the first document.
body = Idx.getAttribute("body-string", docids[0][0])
passage = body[:600]

print("\n==> Retrieval augmented generation <==", flush=True)
print(f'Question: {question}', flush=True)

# This is a very simple prompt template. Fill in your email and code.
prompt = [
    {'role': 'authorize', 'email': 'junchent@andrew.cmu.edu', 'code': 'pqI4'},
    {'role': 'user', 'content': 'TBD'}]

# Do generation with no retrieval.
prompt[1]['content'] = f'question: {question} \n answer: \n'
answer = send_to_llm(llm_address, prompt)
print(f'Answer 1 (no retrieval): {answer}', flush=True)
print(flush=True)

# Do RAG.
prompt[1]['content'] = f'question: {question} \n context: \n {passage}'
answer = send_to_llm(llm_address, prompt)
print(f'Answer 2 (w/retrieval): {answer}', flush=True)
