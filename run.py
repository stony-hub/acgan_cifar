from approaches.acgan import Model


def main():
    model = Model()
    model.train(niter=1)


if __name__ == '__main__':
    main()
