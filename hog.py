import cv2
import numpy as np
import math

cell=8
bins=8
angleperbin = 360 // bins

def extract(img):
        h, w = img.shape
        mag, ang = gradient(img)
        mag = abs(mag)
        cellgradvector = np.zeros((h // cell, w // cell, bins))
        for i in range(cellgradvector.shape[0]):
            for j in range(cellgradvector.shape[1]):
                cellmag = mag[i * cell:(i + 1) * cell,
                                 j * cell:(j + 1) * cell]
                cellang = ang[i * cell:(i + 1) * cell,
                             j * cell:(j + 1) * cell]
                cellgradvector[i][j] = locgradient(cellmag, cellang)

        hog = []
        for i in range(cellgradvector.shape[0] - 1):
            for j in range(cellgradvector.shape[1] - 1):
                vec = []
                vec.extend(cellgradvector[i][j])
                vec.extend(cellgradvector[i][j + 1])
                vec.extend(cellgradvector[i + 1][j])
                vec.extend(cellgradvector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(vec)
                if magnitude != 0:
                    normalize = lambda vec, magnitude: [element / magnitude for element in vec]
                    vec = normalize(vec, magnitude)
                hog.append(vec)
        return hog


def locgradient(cellmag, cellang):
        orns = [0] * bins
        for i in range(cellmag.shape[0]):
            for j in range(cellmag.shape[1]):
                wt = cellmag[i][j]
                ang = cellang[i][j]
                mini, maxi, mod = findbins(ang)
                orns[mini] += (wt * (1 - (mod / angleperbin)))
                orns[maxi] += (wt * (mod / angleperbin))
        return orns

def gradient(img):
        x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        mag = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        ang = cv2.phase(x, y, angleInDegrees=True)
        return mag, ang

def findbins(ang):
        idx = int(ang / angleperbin)
        mod = ang % angleperbin
        if idx == bins:
            return idx - 1, (idx) % bins, mod
        return idx, (idx + 1) % bins, mod


