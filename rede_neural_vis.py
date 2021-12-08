

import matplotlib.pyplot as plt


def ploter(y_test_scaled, predictions, epoc, batch, dias, RMSE):
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(y_test_scaled, label='Contaminações real')
    plt.plot(predictions, label='Contaminações previsão')
    plt.title(
        f" Full previsão\ndias:{dias}, epochs:{epoc}, batch_size:{batch}, RMSE : {RMSE:.2f} ")
    plt.legend()
    plt.show()
    plt.savefig(
        f"NEodias_{dias}_epochs_{epoc}_batch_size_{batch}.png")
