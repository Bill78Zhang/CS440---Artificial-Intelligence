from sklearn.neural_network import MLPClassifier


def import_images(image_file_path):
    data = []
    with open(image_file_path) as f:
        image = []
        counter = 0
        for line in f:
            for c in line:
                image.append(convert_character(c))

            # TODO: Currently only goes to 28. Fix
            if ++counter % 784 != 0:
                continue

            print(image)
            print("\n")
            data.append(image)

            image = []
    return data


def convert_character(c):
    if c == ' ':
        return 0.0
    elif c == '+':
        return 0.5
    else:
        return 1.0


def import_labels(label_file_path):
    labels = []
    with open(label_file_path) as f:
        tmp = f.readlines()

    labels = map(int, tmp)

    return labels


def part_3():
    image_file_path = 'data/trainingimages'
    data = import_images(image_file_path)

    label_file_path = 'data/traininglabels'
    labels = import_labels(label_file_path)

    # for d in data:



if __name__ == "__main__":
    part_3()
