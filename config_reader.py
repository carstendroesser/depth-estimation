def read_config_file(path):
    file = open(path, "r")
    content = []
    for line in file:
        words = line.split()
        for i in range(1, len(words)):
            content.append(words[i])
    file.close()
    return content
