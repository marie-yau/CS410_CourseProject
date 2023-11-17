def get_queries_as_lists_of_lines(lines):
    queries = []

    index = 0
    while index < len(lines):

        if lines[index][0].isdigit():
            current_query = []
            index += 1

            while index < len(lines):
                current_query.append(lines[index])
                index += 1

                if "/" in lines[index]:
                    queries.append(current_query)
                    break

        index += 1

    return queries

def get_queries_as_strings(query_lists):

    query_strings = []

    for query_list in query_lists:
        query_string = " ".join(query_list)
        query_string = query_string.replace("\n", "")
        query_strings.append(query_string)

    return query_strings


with open("../../../datasets/npl/query-text") as f:
    lines = f.readlines()
    queries_list = get_queries_as_lists_of_lines(lines)
    queries_string = get_queries_as_strings(queries_list)

with open("../../../datasets/npl/queries.txt", "a") as f:
    for query_string in queries_string:
        f.write(query_string + "\n")
