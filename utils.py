import numpy as np
import re
from fuzzywuzzy import fuzz
import random
from datetime import datetime



def correct_recalls(i,sigma_t,y_t):
    n_i_plus=sum(1 for j in range(len(sigma_t)) if sigma_t[j] == str(i) and y_t[j] == 1)
    return n_i_plus

def incorrect_recalls(i,sigma_t,y_t):
    n_i_minus=sum(1 for j in range(len(sigma_t)) if sigma_t[j] == str(i) and y_t[j] == 0)
    return n_i_minus

def last_time_of_i(i,sigma_t):
    for j in range(len(sigma_t) - 1, -1, -1):
        if sigma_t[j] == i:
            return j+1
    return 0
        
def find_min(tau,t):
    return min(tau,t)

def update_history(sigma_t,y_t,i,n_y):
    y_updated=[]
    sigma_updated=sigma_t+[i]
    for j in range(len(n_y)):
        y_updated.append(y_t+[n_y[j]])
    return sigma_updated,y_updated


def compute_expected_value(diff,p):
    diff,p=np.array(diff),np.array(p)
    m=np.multiply(diff,p)
    E=sum(m)
    return E

def check_answer(i,response):
    print(i,response)
    response=response.lower()
    i=i.lower()
    print(i,response)
    if i==response:
        return 1
    else:
        return 0
    
def update_sigma_and_y(sigma_t,y_t,i,y):
    sigma_t.append(i)
    y_t.append(y)
    print("func",sigma_t,y_t,len(y_t))
    return sigma_t,y_t

def compute_difference(f1,f2):
    diff=[]
    for j in f2:
        diff.append(j-f1)
    return diff

def generate_shared_theta(lst,n):
    theta = [lst[:] for _ in range(n)]
    return theta

def montecarlo_exp(diff):
    E=np.mean(diff)
    return E

def random_teacher(concepts):
    idx=np.random.randint(0,len(concepts))
    return concepts[idx]

def round_robin_teacher(elements):
    current_index = 0

    def get_next_element():
        nonlocal current_index
        if not elements:
            return None
        element = elements[current_index]
        current_index = (current_index + 1) % len(elements)
        return element

    return get_next_element

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def fuzzy_match(query, targets):
    query = preprocess_text(query)
    scores = []
    
    for idx, target in enumerate(targets):
        target = preprocess_text(target)
        similarity_score = fuzz.ratio(query, target) / 100
        scores.append((idx, target, similarity_score))
    scores.sort(key=lambda x: x[2], reverse=True)
    best_idx, best_target, best_score = scores[0]
    
    return best_score

def extract_lists_from_dict(dictionary):
    concept_numbers = list(dictionary.keys())
    concepts = list(dictionary.values())
    return concept_numbers,concepts

def get_parameters(reward_type, algorithm_type):
    if reward_type=="binary":
        n_y=[0,1]
    elif reward_type=="continuous":
        arr=np.linspace(0,1,1000)
        n_y=np.round(arr,3)
    else:
        return "Invalid reward type"
    return reward_type,algorithm_type

def generate_random_lists(original_list, new_list_size, reward_type):
    shuffled_list = random.sample(original_list, len(original_list))
    random_list = shuffled_list[:new_list_size]

    if len(set(original_list)) > new_list_size:
        remaining_elements = random.sample(set(original_list) - set(random_list), new_list_size - len(random_list))
        random_list.extend(remaining_elements)

    while len(random_list) < new_list_size:
        random_list.append(random.choice(original_list))

    random.shuffle(random_list)

    if reward_type == "binary":
        corresponding_values = [random.randint(0, 1) for _ in range(new_list_size)]
    elif reward_type == "continuous":
        arr = np.linspace(0, 1, new_list_size)
        corresponding_values = np.round(arr, 3)
    else:
        raise ValueError("Invalid reward type. Please specify 'binary' or 'continuous'.")
    return random_list, list(corresponding_values)

def get_flashcard_idx(flashcards,target_answer):
    for i, flashcard in enumerate(flashcards):
        if flashcard["answer"] == target_answer:
            index = i
            break
    return index

def get_filename():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S%m%f")
    return dt_string[:17]
