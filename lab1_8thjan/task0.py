from data import gen_linear_data
from visual import plot_2d_data

def main():
    X,y = gen_linear_data()
    plot_2d_data(X,y, title = "Linear Seprabale Data")

if __name__ == "__main__":
    main()