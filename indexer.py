import math
import re
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Class function to determine the structure of LinkedLists
class Node:
    def __init__(self, doc_id, tf_idf=None):
        self.doc_id = doc_id
        self.tf_idf = tf_idf
        self.skip = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, doc_id, tf_idf=None):
        new_node = Node(doc_id, tf_idf)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def distance(self):
        index_location = self.head
        iterate = 0
        while index_location:
            iterate += 1
            index_location = index_location.next
        return iterate

    def indexing_skip(self):
        measure = self.distance()
        if measure <= 1:
            return
        lenght_of_skip = int(measure ** 0.5)  
        if measure % lenght_of_skip == 0:
            lenght_of_skip -= 1  
        if lenght_of_skip > 0:
            iterate = 0
            index_location = self.head
            last_pointer = None
            while index_location:
                if iterate % lenght_of_skip == 0:
                    if last_pointer:
                        last_pointer.skip = index_location
                    last_pointer = index_location
                    for _ in range(lenght_of_skip):
                        if index_location:
                            index_location = index_location.next
                iterate += 1
                if index_location:
                    index_location = index_location.next

    def streamlined_node(self):
        data = []
        index_location = self.head
        while index_location:
            node_data = {
                'doc_id': index_location.doc_id,
                'tf_idf': index_location.tf_idf,
                'skip': index_location.skip.doc_id if index_location.skip else None
             }
            data.append(node_data)
            index_location = index_location.next
        return data
    
# Processing the text in documents and query.
def text_processing(docs):
    # Lowercase conversion.
    docs = docs.lower()
    # Special Characters removal using regex.
    docs = re.sub(r'[^a-z0-9\s]', '', docs)
    # whitespace removal using regex.
    docs = re.sub(r'\s+', ' ', docs).strip()
    # Tokenzing the document manually based on whitespace.
    tokens = []
    begin = 0
    for j in range(len(docs)):
        if docs[j] == ' ':
            if begin != j:
                tokens.append(docs[begin:j])
            begin = j + 1
    if begin != len(docs):
        tokens.append(docs[begin:])
    # Stemming using porterstemmer.
    # Removing stopwords.
    return [stemmer.stem(token) for token in tokens if token not in stop_words]

# Constructing the Inverted Index.
def ConstructIndex(data_file):
    index = {}
    for doc_id, doc_tokens in enumerate(data_file):
        for token in set(doc_tokens):
            if token not in index:
                index[token] = LinkedList()
            index[token].append(doc_id)
    return index

# Configuring the term frequency and Inverse document frequency within the Linked List.
def Indextf_idf(data_file, index):
    total_docs = len(data_file)
    doc_freq = {term: postings_list.distance() for term, postings_list in index.items()}

    for term, postings_list in index.items():
        df = doc_freq[term]
        idf = math.log10(total_docs / df)
        index_location = postings_list.head
        while index_location:
            doc_id = index_location.doc_id
            term_freq = data_file[doc_id].count(term)
            total_terms_in_doc = len(data_file[doc_id])

            tf = term_freq / total_terms_in_doc
            tf_idf_score = tf * idf
            index_location.tf_idf = tf_idf_score
            index_location = index_location.next

# Document at a Time without using skip function.
# Implementation of the merge algorithm.
# returns sorted list of document ids.
def Document_aat(index, terms):
    if not terms or any(term not in index for term in terms):
        return [], 0

    count_itr = [index[term].head for term in terms]
    output_values = []
    find_value = 0

    while all(it is not None for it in count_itr):
        total_documents = max(it.doc_id for it in count_itr)

        if all(it.doc_id == total_documents for it in count_itr):
            output_values.append(total_documents)
            count_itr = [it.next for it in count_itr]
        else:
            for i, it in enumerate(count_itr):
                while it is not None and it.doc_id < total_documents:
                    it = it.next
                    find_value += 1
                count_itr[i] = it

    return output_values, find_value

# Document at a Time with using skip function.
def Document_aat_skip(index, terms):
    if not terms or any(term not in index for term in terms):
        return [], 0
    count_itr = [index[term].head for term in terms]
    output_values = []
    find_value = 0
    while all(it is not None for it in count_itr):
        total_documents = max(it.doc_id for it in count_itr)
        move_ahead = False
        for i, j in enumerate(count_itr):
            while j is not None and j.doc_id < total_documents:
                if j.skip and j.skip.doc_id <= total_documents:
                    j = j.skip
                else:
                    j = j.next
                find_value += 1
            count_itr[i] = j
            if j is None or j.doc_id > total_documents:
                move_ahead = True
                break
        if not move_ahead:
            output_values.append(total_documents)
            count_itr = [it.next for it in count_itr]
    return output_values, find_value

# Document at a Time with using Tf-idf.
def Documentaat_tfidf(index, terms):
    if not terms or any(term not in index for term in terms):
        return [], 0
    count_itr = [index[term].head for term in terms]
    output_values = []
    find_value = 0
    while all(it is not None for it in count_itr):
        total_documents = max(it.doc_id for it in count_itr)
        if all(it.doc_id == total_documents for it in count_itr):
            tf_idf_sum = sum(it.tf_idf for it in count_itr)
            output_values.append((total_documents, tf_idf_sum))
            count_itr = [it.next for it in count_itr]
        else:
            for i, j in enumerate(count_itr):
                while j is not None and j.doc_id < total_documents:
                    j = j.next
                    find_value += 1
                count_itr[i] = j
    output_values.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in output_values], find_value

# Document at a Time with using Tf-idf and skip function.
def Documentaat_tfidf_skip(index, terms):
    if not terms or any(term not in index for term in terms):
        return [], 0
    count_itr = [index[term].head for term in terms]
    output_values = []
    find_value = 0
    while all(j is not None for j in count_itr):
        total_documents = max(j.doc_id for j in count_itr)
        if all(j.doc_id == total_documents for j in count_itr):
            tf_idf_sum = sum(j.tf_idf for j in count_itr)
            output_values.append((total_documents, tf_idf_sum))
            count_itr = [j.next for j in count_itr]
        else:
            for i, j in enumerate(count_itr):
                while j is not None and j.doc_id < total_documents:
                    # Use skip pointer if possible
                    if j.skip and j.skip.doc_id <= total_documents:
                        j = j.skip
                    else:
                        j = j.next
                    find_value += 1
                count_itr[i] = j
    output_values.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in output_values], find_value

# Traverse the nodes for the LinkedList.
def traverse_node(lst_LL):
    index_location = lst_LL.head
    while index_location:
        yield index_location
        index_location = index_location.next
