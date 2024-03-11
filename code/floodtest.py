import numpy as np
import floodfill

def main():
    x = np.array([[0,0,1,1,0],[0,0,1,1,0],[0,1,1,1,0]]).astype(np.uint8)
    y = floodfill.flood(x,0,0)
    print(x)

if __name__ == '__main__':
    main()