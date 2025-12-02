import numpy as np
from typing import Tuple

def nearestLink(Pi: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find the nearest link to a given point p0 from a set of points Pi.
    Args:
        Pi (np.ndarray): A 2D array where each row represents a point (link).
        p0 (np.ndarray): A 1D array representing the point to find the nearest link to.
    Returns:
        Tuple[np.ndarray, int]: A tuple containing the nearest link point and its index.
    1. Piの各行はリンクを表す2D配列。
    2. p0は1D配列で、最も近いリンクを見つけるための点を表す。
    3. 戻り値は、最も近いリンクの点とそのインデックスを含むタプル。
    """
    if len(np.shape(p0)) == 1:
        p0 = np.reshape(p0, [1, len(p0)])
    if Pi.shape[0] == 1:
        # 行数が1の場合、その点を返す
        return Pi[0], int(0)
    Pr = Pi[1:, :] - Pi[:-1, :]
    P = np.ones([len(Pr), 1]) @ p0 - Pi[:-1, :]
    a = sum((Pr * P).T)
    b = sum((Pr * Pr).T)
    klist = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    klim = np.clip(klist, 0, 1)
    Pn = np.diag(klim) @ Pr
    PnP = Pn - P
    l2 = sum((PnP * PnP).T)
    idx = int(np.argmin(l2))
    pout = Pn[idx, :] + Pi[idx, :]
    return pout, idx

if __name__ == "__main__":
    poslist = np.array([[1,1],[1,2],[2,2]])
    pos = np.array([[1.1,2.1]])
    print(nearestLink(poslist,pos))
