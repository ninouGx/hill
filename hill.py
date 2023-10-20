import numpy as np
import math as m

ALPHABET = "abcdefghijklmnopqrstuvwxyz :"
CODED_MSG = "Cryptage et décryptage : communiquer en toute sécurité"
CIPHER_KEY = np.array([[9, 6], [2, 7]])
CIPHER_KEY_3 = np.array([[4,1,5],[11,19,1], [2,3,2]])
CIPHER_KEY_2 = np.array([[2,2,1],[2,1,1], [1,0,1]])

"""
Alphabet de 26 lettres minuscules (on ne distinguera pas les lettres majuscules des lettres
minuscules et on considère le message comme non accentué), l'espace et les « : » 
"""
def transform_to_valid_str(string: str = CODED_MSG) -> str:
    """
    Transforme une chaine de caractères en une chaine de caractères valide
    """
    string = string.lower()
    string = string.replace("é", "e")
    string = string.replace("è", "e")
    string = string.replace("ê", "e")
    string = string.replace("ë", "e")
    string = string.replace("à", "a")
    string = string.replace("â", "a")
    string = string.replace("ä", "a")
    string = string.replace("ù", "u")
    string = string.replace("û", "u")
    string = string.replace("ü", "u")
    string = string.replace("î", "i")
    string = string.replace("ï", "i")
    string = string.replace("ô", "o")
    string = string.replace("ö", "o")
    string = string.replace("ç", "c")
    string = string.replace("œ", "oe")
    string = string.replace("æ", "ae")
    return string

def ask_for_cipher_key(size: int = 2) -> np.ndarray:
    """
    Demande à l'utilisateur de rentrer une clé de chiffrement de taille size
    """
    print("Votre clé sera une matrice carré de taille {}".format(size))
    cipher_key = []
    for i in range(size):
        cipher_key.append([])
        for j in range(size):
            cipher_key[i].append(int(input("Veuillez rentrer la valeur de la clé de chiffrement en [ {} , {} ] : ".format(i+1, j+1))))
    return np.array(cipher_key)
    

def get_index_from_char(char: str) -> int:
    """
    Retourne l'index d'un caractère dans l'alphabet
    """
    return ALPHABET.index(char)

def get_char_from_index(index: int) -> str:
    """
    Retourne le caractère correspondant à l'index dans l'alphabet
    """
    return ALPHABET[index]

def get_list_index_from_str(string: str) -> list:
    """
    Retourne une liste d'index correspondant à chaque caractère de la chaine de caractères
    """
    return [get_index_from_char(char) for char in string]

def get_str_from_list_index(list_index: list) -> str:
    """
    Retourne une chaine de caractères correspondant à chaque index de la liste
    """
    return "".join([get_char_from_index(index) for index in list_index])

def get_ngram_from_list_index(list_index: list, n: int) -> list:
    """
    Retourne une liste de n-grammes (np) à partir d'une liste d'index
    si la liste d'index n'est pas divisible par n, on complète avec des index aléatoires
    "hello" -> [array([7, 4]), array([11, 11]), array([14, random(0, 27)]) 
    """
    list_ngram = []
    for i in range(0, len(list_index) - n + 1, n):
        list_ngram.append(np.array(list_index[i:i+n]))
    if len(list_index) % n != 0:
        random_index = np.random.randint(0, len(ALPHABET), n - len(list_index) % n)
        list_ngram.append(np.array(list_index[-(len(list_index) % n):] + list(random_index)))
    return list_ngram

def get_matrix_determinant(matrix: np.ndarray) -> int:
    return int(np.linalg.det(matrix))

def is_matrix_det_first_size_alphabet(matrix: np.ndarray) -> bool:
    """
    Retourne True si le déterminant de la matrice est un multiple de la taille de l'alphabet
    """
    return m.gcd(get_matrix_determinant(matrix), len(ALPHABET)) == 1

def get_calcul_matrix(list_index: list, matrix: np.ndarray = CIPHER_KEY) -> np.ndarray:
    """
    Retourne la matrice résultante du calcul de la matrice passée en paramètre avec la liste d'index
    """
    coded_list = []
    for ngram in list_index:
        coded_list.append(np.dot(matrix, ngram) % len(ALPHABET))
    return np.array(coded_list)

def get_cofactor_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Retourne la matrice des cofacteurs de la matrice passée en paramètre
    """
    return np.linalg.inv(matrix).T * get_matrix_determinant(matrix)

def get_transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Retourne la matrice transposée de la matrice passée en paramètre en int
    """
    return matrix.T

def get_inverse_mod(nb: int, mod: int = len(ALPHABET)) -> int:
    """
    Retourne l'inverse de nb modulo mod
    """
    return int(pow(nb, -1, mod))   

def size2():
    #cipher_key = ask_for_cipher_key()
    cipher_key = CIPHER_KEY
    #cipher_key = np.array([[15,37],[41,118]])
    print(cipher_key)
    
    determinant = get_matrix_determinant(cipher_key)
    print(determinant)
    
    inv_mod = get_inverse_mod(determinant)
    print(inv_mod)
    
    print("Le déterminant de la matrice est : {}".format(get_matrix_determinant(cipher_key)))
    print("Le déterminant de la matrice est premier avec {} ?".format(len(ALPHABET)), is_matrix_det_first_size_alphabet(cipher_key))
    print("L'inverse modulaire de {} modulo {} est : {}".format(get_matrix_determinant(cipher_key), len(ALPHABET), get_inverse_mod(get_matrix_determinant(cipher_key))))
    
    proper_msg = transform_to_valid_str("hello")
    proper_msg = transform_to_valid_str()
    print(proper_msg)
    
    list_index = get_list_index_from_str(proper_msg)
    print(list_index)
    
    ngram = get_ngram_from_list_index(list_index, 2)
    print(ngram)
    
    coded_ngram = get_calcul_matrix(ngram)
    print(coded_ngram)
    
    coded_str = get_str_from_list_index(coded_ngram.flatten())
    print(coded_str)
    
    transpose_cofactor_matrix = get_transpose_matrix(get_cofactor_matrix(cipher_key))
    print(transpose_cofactor_matrix)
    
    reverse_cipher_key = (inv_mod * transpose_cofactor_matrix) % len(ALPHABET)
    print(reverse_cipher_key)
    
    decoded_ngram = get_calcul_matrix(coded_ngram, reverse_cipher_key)
    print(decoded_ngram)
    
    decoded_str = get_str_from_list_index(decoded_ngram.flatten().astype(int))
    print(decoded_str)

def size3():
        #cipher_key = ask_for_cipher_key()
    cipher_key = CIPHER_KEY_2
    #cipher_key = np.array([[15,37],[41,118]])
    print(cipher_key)
    
    determinant = get_matrix_determinant(cipher_key)
    print(determinant)
    
    inv_mod = get_inverse_mod(determinant)
    print(inv_mod)
    
    print("Le déterminant de la matrice est : {}".format(get_matrix_determinant(cipher_key)))
    print("Le déterminant de la matrice est premier avec {} ?".format(len(ALPHABET)), is_matrix_det_first_size_alphabet(cipher_key))
    print("L'inverse modulaire de {} modulo {} est : {}".format(get_matrix_determinant(cipher_key), len(ALPHABET), get_inverse_mod(get_matrix_determinant(cipher_key))))
    
    proper_msg = transform_to_valid_str("hello")
    proper_msg = transform_to_valid_str()
    print(proper_msg)
    
    list_index = get_list_index_from_str(proper_msg)
    print(list_index)
    
    ngram = get_ngram_from_list_index(list_index, 3)
    print(ngram)
    
    coded_ngram = get_calcul_matrix(ngram, cipher_key)
    print(coded_ngram)
    
    coded_str = get_str_from_list_index(coded_ngram.flatten())
    print(coded_str)
    
    transpose_cofactor_matrix = get_transpose_matrix(get_cofactor_matrix(cipher_key))
    print(transpose_cofactor_matrix)
    
    reverse_cipher_key = (inv_mod * transpose_cofactor_matrix) % len(ALPHABET)
    print(reverse_cipher_key)
    
    decoded_ngram = get_calcul_matrix(coded_ngram, reverse_cipher_key)
    print(decoded_ngram)
    
    decoded_str = get_str_from_list_index(decoded_ngram.flatten().astype(int))
    print(decoded_str)

def main():
    size3()
    

if __name__ == "__main__":
    main()