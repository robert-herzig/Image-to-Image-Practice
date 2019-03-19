import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict

class CSVPlotter:
    def __init__(self, filepath, imagepath):
        print("INITIALIZED CSV_PLOTTER ON FILE " + str(filepath))
        self.filepath = filepath
        self.imagepath = imagepath

    def plot_csv_to_image(self, plot_discriminator):
        with open(self.filepath, 'r') as csvfile:
            epoch = []
            g = []
            val = []
            d = []

            cur_plots = csv.reader(csvfile, delimiter=',')
            row_count = 0
            for row in cur_plots:
                epoch.append(row_count)
                row_count += 1
                g.append(float(row[0]))
                val.append(float(row[1]))
                if plot_discriminator:
                    d.append(float(row[2]))

            plt.plot(epoch, g, label = "Train-Loss")
            plt.plot(epoch, val, label = "Val-Loss")
            if plot_discriminator:
                plt.plot(epoch, d, label="D-Loss")

            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.grid(True)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            # plt.legend()
            plt.savefig(self.imagepath)


