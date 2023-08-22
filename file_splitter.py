def split(path, count):
    with open(path, 'r') as file:
        lines = file.readlines()
        start = 0
        delta = (len(lines) + 1) // count
        end = delta
        for i in range(count):
            contents = ''
            ls = lines[start:end]
            for l in ls:
                contents += l
            with open(path + str(i), 'w') as dest:
                dest.write(contents)
            start = end
            end += delta
