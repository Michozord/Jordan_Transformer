"""
Moduł pomocniczy do generowania macierzy testowych używanych w eksperymentach.

Zawiera funkcje do tworzenia losowych macierzy o określonej strukturze oraz
funkcję do generowania zestawu testowego (macierze + etykiety). Opisy parametrów
funkcji i ich zachowania znajdują się w docstringach przy poszczególnych
funkcjach.
"""

import numpy as np
import scipy
from math import inf

def generate_matrix(d, block_size, mode, eps=None, lam=1, value_range=None, schur=False):
    """
    Generuje macierz testową o rozmiarze `d x d` z kontrolowaną strukturą.

    Parametry:
    - d (int): wymiar macierzy (liczba wierszy/kolumn).
    - block_size (int): liczba pozycji na nadprzekątnej, które zostaną ustawione
        na 1 (losowo wybrane indeksy). Używane do budowy macierzy Jordanowskiej.
    - mode (str): tryb generacji macierzy pomocniczej `S`. Obsługiwane wartości:
        "random", "int", "upper", "lower", "ortho". Określa, jakiego rodzaju
        macierz S zostanie wygenerowana (np. ortogonalna, trójkątna itp.).
    - eps (float|None): jeśli nie None, dodaje małą losową perturbację do macierzy
        J (skaluje losowymi wartościami z [0, eps)). Przydatne do testów stabilności.
    - lam (float): wartość dodawana do elementów diagonalnych macierzy J (domyślnie 1).
    - value_range (int|float|None): skala wartości używana przy generacji macierzy S;
        jeśli None, wartość dobierana jest zależnie od `mode`.
    - schur (bool): jeśli True, zwraca postać Schura macierzy X zamiast X bezpośrednio.

    Zwraca:
    - X (np.ndarray): macierz wynikowa rozmiaru (d, d). Jeśli `schur`==True, zwracana
        jest macierz otrzymana z dekompozycji Schura.

    Uwaga:
    Funkcja tworzy macierz J (prawie Jordanowską) z nadprzekątną ustawioną zgodnie
    z `block_size`, a następnie wykonuje podobieństwo X = S @ J @ S^{-1}, gdzie S
    jest generowane według `mode`.
    """
    indexes = np.random.choice(d-1, size=block_size, replace=False)
    # indexes = list(range(block_size))
    super_diag = np.zeros(d-1)
    for index in indexes:
        super_diag[index] = 1
    J = lam * np.eye(d) + np.diag(super_diag, k=1)
    if eps is not None:
        J += eps * np.random.rand(d, d)
    if value_range is None:
        match mode:
            case "random" | "upper" | "ortho" | "lower":
                value_range = 1
            case "int":
                value_range = 100
            case _:
                raise RuntimeError(f"Mode {mode} is not supported")

    def generate_S():
        while True:
            match mode:
                case "random":
                    S = np.random.rand(d, d) * value_range
                case "int":
                    S = np.random.randint(0, value_range, size=(d, d))
                case "upper":
                    S = np.triu(np.random.rand(d, d)) * value_range
                case "lower":
                    S = np.tril(np.random.rand(d, d)) * value_range
                case "ortho":
                    A = np.random.rand(d, d)
                    Q, _ = np.linalg.qr(A)
                    S = Q
            if abs(np.linalg.cond(S)) < 1e5:
                return S

    S = generate_S()
    X = S @ J @ np.linalg.inv(S)
    # X = X / np.linalg.norm(X, ord="fro")
    if schur:
        return scipy.linalg.schur(X)[0]
    else:
        return X


def generate_testset(d, size_per_class, mode="random", eps=None, schur=False):
    """
    Generuje zbiór testowy macierzy wraz z etykietami klas.

    Parametry:
    - d (int): wymiar macierzy (liczba klas również wynika z `d`).
    - size_per_class (int): liczba przykładów (macierzy) generowanych dla każdej klasy.
    - mode (str): tryb przekazywany dalej do `generate_matrix` (jak wyżej).
    - eps (float|None): opcjonalna perturbacja przekazywana do `generate_matrix`.
    - schur (bool): jeśli True, każda macierz będzie zwrócona w postaci Schura.

    Zwraca:
    - X (np.ndarray): tablica kształtu `(size_per_class * d, d, d)`, gdzie pierwszym
      wymiarem indeksujemy próbki.
    - y (list[int]): lista etykiet (0..d-1) odpowiadająca klasom każdej macierzy w X.

    Zachowanie:
    Dla każdej etykiety `label` od 0 do `d-1` generowane jest `size_per_class`
    macierzy przy użyciu `generate_matrix(d, label, ...)`. Etykieta przypisywana jest
    zgodnie z wartością `label` (może reprezentować np. rozmiar bloku/Jordanowską pozycję).
    """
    X = np.ndarray(shape=(size_per_class * d, d, d))
    y = []

    idx = 0
    for label in range(d):
        for _ in range(size_per_class):
            X[idx] = generate_matrix(d, label, mode, eps=eps, schur=schur)
            idx += 1
            y.append(label)

    return X, y
