import json
import os

from scipy.optimize import direct


# Funció per guardar un diccionari en un fitxer .txt
def save_data(name_file, dictionary):
    directory = name_file.split('/')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    name_file = name_file + ".txt"
    # Si el fitxer ja està creat l'eliminem
    if os.path.exists(name_file):
        os.remove(name_file)
    # Guardem les dades al fitxer
    with open(name_file, 'w') as data:
        data.write(json.dumps(dictionary))


# Funció per carregar un diccionari des d'un fitxer .txt
def load_data(name_file):
    name_file = name_file + ".txt"
    with open(name_file) as fW:
        dataW = fW.read()
    return json.loads(dataW)