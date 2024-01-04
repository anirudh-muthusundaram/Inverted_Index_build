import pandas as pd
import time
import json
from indexer import text_processing, ConstructIndex, Document_aat, Document_aat_skip, traverse_node, Indextf_idf, Documentaat_tfidf, Documentaat_tfidf_skip

# Read the input_corpus.txt.
def data_file(data_input):
    with open(data_input, 'r', encoding='utf-8') as data:
        return [text_processing(row.strip()) for row in data]

# Construct of Posting list without skip function.
def PostList(postings):
    id_iteration = []
    pointer_location = postings.head
    while pointer_location:
        document_num = pointer_location.doc_id  
        id_iteration.append(document_num)
        pointer_location = pointer_location.next
    return id_iteration

# Construct of Posting list with skip function.
def PostList_skip(postings):
    data_list = []
    pointer_location = postings.head
    if not pointer_location or not pointer_location.skip:
        return data_list
    while pointer_location:
        data_list.append(pointer_location.doc_id)
        if pointer_location.skip:
            data_list.append(pointer_location.skip.doc_id)
            pointer_location = pointer_location.skip
        else:
            break
    return data_list


if __name__ == "__main__":
    # Locate the file to read.
    data_document = data_file("C:\College\IR\Project 2\src\Data\input_corpus.txt")
    # Creating the Inverted Index.
    index = ConstructIndex(data_document)
    outset = time.time()
    # Tf-idf computation.
    Indextf_idf(data_document, index)
    # Adding Skip pointers to Inverted Index
    for term, final_pos in index.items():
        final_pos.indexing_skip()
    # Query verification.
    queries = ["from an epidemic to a pandemic", "is hydroxychloroquine effective?", "the novel coronavirus"]
    # output_json container.
    response = {}
    # Computation of daatAnd and daatAndSkip.
    for type in ["daatAnd", "daatAndSkip"]:
        output_query = {}
        for count in queries:
            word_qry = text_processing(count)
            if type == "daatAnd":
                output_ids, value_identify = Document_aat(index, word_qry)
            else:
                output_ids, value_identify = Document_aat_skip(index, word_qry)
            
            output_query[count] = {
                "num_comparisons": value_identify,
                "num_docs": len(output_ids),
                "results": output_ids
            }
        response[type] = output_query

    # Computation of daatAndTfIdf and daatAndSkipTfIdf.
    for type in ["daatAndTfIdf", "daatAndSkipTfIdf"]:
        output_query = {}
        for i in queries:
            word_qry = text_processing(i)
            if type == "daatAndTfIdf":
                output_ids, value_identify = Documentaat_tfidf(index, word_qry)
            else:
                output_ids, value_identify = Documentaat_tfidf_skip(index, word_qry)
            output_query[i] = {
                "num_comparisons": value_identify,
                "num_docs": len(output_ids),
                "results": output_ids
            }
        response[type] = output_query

    # Construct the structure of output_json
    response["postingsList"] = {term: PostList(final_pos) for term, final_pos in index.items()}
    response["postingsListSkip"] = {term: PostList_skip(final_pos) for term, final_pos in index.items()}
    response["time_taken"] = time.time() - outset

    # Output json
    verify_json = json.dumps({"Response": response}, indent=4)
    print(verify_json)
