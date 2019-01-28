import matplotlib.pyplot as plt
import csv
import numpy as np

class CSVPlotter:
    def __init__(self, filepath, imagepath):
        print("INITIALIZED CSV_PLOTTER ON FILE " + str(filepath))
        self.filepath = filepath
        self.imagepath = imagepath

    def plot_csv_to_image(self):
        with open(self.filepath, 'r') as csvfile:
            epoch = []
            g = []
            d = []

            cur_plots = csv.reader(csvfile, delimiter=',')
            row_count = 0
            for row in cur_plots:
                epoch.append(row_count)
                row_count += 1
                g.append(float(row[0]))
                # d.append(float(row[1]))

            plt.plot(epoch, g, label = "G-Loss")
            # plt.plot(epoch, d, label = "D-Loss")

            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.grid(True)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=1, mode="expand", borderaxespad=0.)
            plt.savefig(self.imagepath)


