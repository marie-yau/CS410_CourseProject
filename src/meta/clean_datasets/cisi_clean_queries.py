def get_queries_as_lists_of_lines(lines):
    queries = []

    index = 0
    while index < len(lines):

        if lines[index].startswith(".I"):
            current_query = []

            while index < len(lines):
                current_query.append(lines[index])
                index += 1

                if index == len(lines) or lines[index].startswith(".I"):
                    queries.append(current_query)
                    break
        if index < len(lines) and not lines[index].startswith(".I"):
            index += 1

    return queries

def get_queries_as_strings(doc_lists):

    query_strings = []

    for doc_list in doc_lists:
        query_string = " ".join(doc_list)
        query_string = query_string.replace(".I", "")
        query_string = query_string.replace(".T", "")
        query_string = query_string.replace(".A", "")
        query_string = query_string.replace(".W", "")
        query_string = query_string.replace("\n", "")
        query_string = query_string.lstrip('0123456789 ')
        query_strings.append(query_string)

    return query_strings


with open("../../../datasets/cisi/CISI.QRY") as f:
    lines = f.readlines()
    query_list = get_queries_as_lists_of_lines(lines)
    query_strings = get_queries_as_strings(query_list)
    print(query_strings)

with open("../../../datasets/cisi/queries.txt", "a") as f:
    for query_string in query_strings:
        f.write(query_string + "\n")
