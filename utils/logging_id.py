import os
import os.path as path

STORE_PATH = ".data"


def data_path(string):
    return path.join(STORE_PATH, string + "_id")


def get_id(string):
    id_path = data_path(string)
    if not path.exists(id_path):
        reset_id(string)
    with open(id_path, "r") as f:
        my_id = int(next(f))
    return string + str(my_id)


def update_id(string):
    id_path = data_path(string)

    if not path.exists(id_path):
        reset_id(string)
        my_id = 1
    else:
        with open(id_path, "r") as f:
            my_id = int(next(f)) + 1
        with open(id_path, "w") as f:
            f.write(str(my_id))
    return string + str(my_id)


def reset_id(string):
    id_path = data_path(string)

    with open(id_path, "w") as f:
        f.write("0")
    r = "0"
    return r


if not os.path.exists(STORE_PATH):
    os.mkdir(STORE_PATH)
    reset_id("experiment")
