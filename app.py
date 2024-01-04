from flask import Flask, request, jsonify
import pandas as pd
import json
import time
from indexer import text_processing, ConstructIndex, Document_aat, Document_aat_skip, Indextf_idf, Documentaat_tfidf, Documentaat_tfidf_skip
from main import data_file, PostList, PostList_skip

app = Flask(__name__)

# Assigning the route to execute the query as per instruction.
# API endpoint for the processing.
@app.route('/execute_query', methods=['POST'])
def execute_query():
    input_query = request.get_json()
    queries = input_query.get("queries", [])
    response = {}  
    # Computation of daatAnd and daatAndSkip.
    for type in ["daatAnd", "daatAndSkip"]:
        output_query = {}
        # Processing the input queries.
        for count in queries:
            query_terms = text_processing(count)
            if type == "daatAnd":
                output_docsID, value_identify = Document_aat(index, query_terms)
            else:
                output_docsID, value_identify = Document_aat_skip(index, query_terms)
            output_query[count] = {
                "num_comparisons": value_identify,
                "num_docs": len(output_docsID),
                "results": output_docsID
            }
        response[type] = output_query
    # Computation of daatAndTfIdf and daatAndSkipTfIdf.
    for type in ["daatAndTfIdf", "daatAndSkipTfIdf"]:
        output_query = {}
        for count in queries:
            query_terms = text_processing(count)
            if type == "daatAndTfIdf":
                output_docsID, value_identify = Documentaat_tfidf(index, query_terms)
            else:
                output_docsID, value_identify = Documentaat_tfidf_skip(index, query_terms)
            output_query[count] = {
                "num_comparisons": value_identify,
                "num_docs": len(output_docsID),
                "results": output_docsID
            }
        response[type] = output_query
    # Construct the structure of output_json
    response["postingsList"] = {term: PostList(postings_list) for term, postings_list in index.items()}
    response["postingsListSkip"] = {term: PostList_skip(postings_list) for term, postings_list in index.items()}
    response["time_taken"] = time.time() - outset
    return jsonify({"Response": response})

if __name__ == '__main__':

    # Locate the file to read.
    data_document = data_file("C:\College\IR\Project 2\src\Data\input_corpus.txt")
    # Determining the timetaken.
    outset = time.time()
    # Creating the Inverted Index.
    index = ConstructIndex(data_document)
    # Tf-idf computation.
    Indextf_idf(data_document, index)
    # Skip function integration.
    for term, postings_list in index.items():
        postings_list.indexing_skip()
    # Hosting the application based on the required port.
    app.run(host='0.0.0.0', port=9999)
