import glob
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QAbstractTableModel, QDir
from PyQt5.QtGui import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QPushButton,
    QTableView,
    QToolBar,
)

MULTI_ROOT = "multirun.yaml"
SOURCE = "results.csv"
XCOL = "env_step"
YCOL = "episode_reward"


class ExperimentsModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.data = []
        for path in [
            path.replace("/{}".format(SOURCE), "/.hydra/config.yaml") for path in data
        ]:
            config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            entry = [config["algorithm"]["name"]]
            entry = entry + [config["experiment"]]
            entry = entry + [
                config["env"] if "env" in config else config["overrides"]["env"]
            ]
            entry = entry + [config["seed"]]
            self.data.append(entry)
        self._headers = ["Algorithm", "Experiment", "Environment", "Seed"]

    def rowCount(self, parent=None):
        return len(self.data)

    def columnCount(self, parent=None):
        return len(self.data[0]) if self.rowCount(parent) > 0 else 0

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self.data[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._headers[col]
        return None


class BasicTrainingResultsWindow(QMainWindow):
    def __init__(self, experiment_root):
        super(BasicTrainingResultsWindow, self).__init__()

        self.experiment_root = experiment_root
        if self.experiment_root[-1] != "/":
            self.experiment_root = self.experiment_root + "/"

        self.load_experiments()

        self.chart = Figure()
        self.axes = self.chart.add_subplot(111)
        self.graphWidget = FigureCanvas(self.chart)
        self.setCentralWidget(self.graphWidget)

        self.logYAxisCheckbox = QCheckBox("Log Scale (Y Axis)")
        self.logYAxisCheckbox.stateChanged.connect(self.onChangeScale)

        self.displayAsDistributionCheckbox = QCheckBox("Display As Distribution")
        self.displayAsDistributionCheckbox.stateChanged.connect(
            self.onChangeDisplayAsDistribution
        )
        self.displayAsDistributionCheckbox.setEnabled(False)

        self.saveFigureButton = QPushButton("&Save Figure")
        self.saveFigureButton.clicked.connect(self.onSaveFigure)

        self.optionsToolBar = QToolBar(self)
        self.optionsToolBar.addWidget(self.saveFigureButton)
        self.optionsToolBar.addWidget(self.logYAxisCheckbox)
        self.optionsToolBar.addWidget(self.displayAsDistributionCheckbox)
        self.addToolBar(self.optionsToolBar)

        self.resultsWidget = QDockWidget("Experiments", self)
        self.experimentTable = ExperimentsModel(self.experiment_results, self)
        self.tableView = QTableView()
        self.tableView.setModel(self.experimentTable)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.selectionModel().selectionChanged.connect(
            self.onExperimentsSelectionChanged
        )
        self.resultsWidget.setWidget(self.tableView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.resultsWidget)

    def load_experiments(self):
        self.experiment_results = glob.glob(
            self.experiment_root + "**/{}".format(SOURCE), recursive=True
        )
        self.multirun_results = glob.glob(
            self.experiment_root + "**/{}".format(MULTI_ROOT), recursive=True
        )
        self.experiment_names = []
        for path in self.experiment_results:
            name = path.replace(self.experiment_root, "").replace(
                "/{}".format(SOURCE), ""
            )
            self.experiment_names.append(name)

    def selectedMatchingSequences(self):
        selection = self.tableView.selectionModel()

        if selection.hasSelection():
            if len(selection.selectedRows()) == 1:
                return False

            first_length = None
            index_series = None
            for rowIndex in [row.row() for row in selection.selectedRows()]:
                result = pd.read_csv(self.experiment_results[rowIndex])
                first_length = (
                    first_length if first_length is not None else result[XCOL].size
                )
                index_series = (
                    index_series if index_series is not None else result[XCOL]
                )
                if (first_length != result[XCOL].size) or (
                    not index_series.equals(other=result[XCOL])
                ):
                    return False

            return True

        return False

    def onSaveFigure(self, checked):
        fname, _ = QFileDialog.getSaveFileName(
            self, caption="Save figure", filter="Figures (*.png *.pdf)"
        )
        if len(fname) > 0:
            self.chart.savefig(fname)

    def onChangeScale(self, state):
        self.onExperimentsSelectionChanged()

    def onFilterOutliers(self, state):
        self.onExperimentsSelectionChanged()

    def onChangeDisplayAsDistribution(self, state):
        self.onExperimentsSelectionChanged()

    def onExperimentsSelectionChanged(self, sel1=None, sel2=None):
        selection = self.tableView.selectionModel()

        if selection.hasSelection():
            self.axes.cla()

            self.displayAsDistributionCheckbox.setEnabled(
                self.selectedMatchingSequences()
            )
            displayAsDistribution = (
                self.displayAsDistributionCheckbox.isEnabled()
                and self.displayAsDistributionCheckbox.checkState()
            )

            series = []
            for rowIndex in [row.row() for row in selection.selectedRows()]:
                result = pd.read_csv(self.experiment_results[rowIndex])

                if displayAsDistribution:
                    if len(series) == 0:
                        series.append(result[XCOL])
                    series.append(result[YCOL])
                else:
                    self.axes.plot(
                        result[XCOL],
                        result[YCOL],
                        label=self.experiment_names[rowIndex],
                    )

            if displayAsDistribution:
                time_series = series[0]
                data_series = series[1:]

                df = pd.concat(data_series, axis=1)
                mean_series = df.mean(axis=1)
                var_series = df.std(axis=1)

                mean_line = self.axes.plot(
                    time_series, mean_series, label=self.experiment_results[rowIndex]
                )
                self.axes.fill_between(
                    time_series,
                    mean_series + var_series,
                    mean_series - var_series,
                    color=mean_line[0].get_color(),
                    alpha=0.25,
                )

            if self.logYAxisCheckbox.checkState():
                self.axes.semilogy()

            self.axes.legend()
            self.graphWidget.draw()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize training results")
    argparser.add_argument(
        "--experiments",
        type=str,
        default=QDir.currentPath() + "/exp/",
        help="The path to the experiments folder",
    )
    args = argparser.parse_args()

    if not os.path.exists(args.experiments):
        print("Path " + args.experiments + " does not exist.")
        exit(-1)

    # create the application and the main window
    app = QApplication(sys.argv)
    window = BasicTrainingResultsWindow(args.experiments)

    # run
    window.showMaximized()
    app.exec_()
