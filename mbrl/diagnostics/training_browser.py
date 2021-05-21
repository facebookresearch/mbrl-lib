# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import glob
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QAbstractTableModel, QDir, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QPushButton,
    QTableView,
    QToolBar,
)

MULTI_ROOT = "multirun.yaml"
SOURCE = "results.csv"
XCOL = "env_step"
YCOL = "episode_reward"

user_name_dict = {"x_label": "steps", "y_label": "episode reward"}


class ExperimentsModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, distributionCheckbox, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.distributionCheckbox = distributionCheckbox
        self.reloadExperiments(data)

        self._headers = ["Algorithm", "Experiment", "Environment", "Seed"]

    def reloadExperiments(self, data):
        self.data = []
        for path in data:
            if path.endswith(SOURCE):
                config = yaml.load(
                    open(
                        path.replace("/{}".format(SOURCE), "/.hydra/config.yaml"), "r"
                    ),
                    Loader=yaml.FullLoader,
                )
            elif path.endswith(MULTI_ROOT):
                config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
            entry = [config["algorithm"]["name"]]
            entry = entry + [config["experiment"]]
            entry = entry + [
                config["env"] if "env" in config else config["overrides"]["env"]
            ]
            entry = entry + [config["seed"]]
            self.data.append(entry)

        self.modelReset.emit()

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


class GraphLabels(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, names, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.reloadExperiments(names)

        self._headers = ["Algorithm", "Experiment", "Environment", "Seed"]

    def reloadExperiments(self, names):
        self.names = copy.deepcopy(names)
        self.names = self.names + ["x_label", "y_label"]

        self.modelReset.emit()

    def rowCount(self, parent=None):
        return len(self.names)

    def columnCount(self, parent=None):
        return 1

    def flags(self, index):
        if not index.isValid():
            return 0

        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role):
        global user_name_dict

        if role == Qt.EditRole:
            row = index.row()
            key = (
                self.names[index.row()]
                if (row < (self.rowCount() - 2))
                else ("x_label" if (row < (self.rowCount() - 1)) else "y_label")
            )
            user_name_dict[key] = value

            self.dataChanged.emit(index, index, [Qt.EditRole])

            return True

        return False

    def data(self, index, role=Qt.DisplayRole):
        global user_name_dict

        if index.isValid():
            if role == Qt.DisplayRole:
                return str(user_name_dict[self.names[index.row()]])
        return None

    def headerData(self, row, orientation, role):
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (
                f"Experiment #{row+1}"
                if (row < (self.rowCount() - 2))
                else ("X Label" if (row < (self.rowCount() - 1)) else "Y Label")
            )
        return None


class BasicTrainingResultsWindow(QMainWindow):
    def __init__(self, experiment_root):
        super(BasicTrainingResultsWindow, self).__init__()

        self.experiment_root = experiment_root
        if self.experiment_root[-1] != "/":
            self.experiment_root = self.experiment_root + "/"

        self.chart = Figure()
        self.axes = self.chart.add_subplot(111)
        self.graphWidget = FigureCanvas(self.chart)
        self.setCentralWidget(self.graphWidget)

        self.logYAxisCheckbox = QCheckBox("Log Scale (Y Axis)")
        self.logYAxisCheckbox.stateChanged.connect(self.onChangeScale)

        self.displayAsDistributionCheckbox = QCheckBox("Aggregate Results (Mean/Std)")
        self.displayAsDistributionCheckbox.setChecked(True)

        self.saveFigureButton = QPushButton("&Save Figure")
        self.saveFigureButton.clicked.connect(self.onSaveFigure)

        self.optionsToolBar = QToolBar(self)
        self.optionsToolBar.addWidget(self.saveFigureButton)
        self.optionsToolBar.addWidget(self.logYAxisCheckbox)
        self.optionsToolBar.addWidget(self.displayAsDistributionCheckbox)
        self.addToolBar(self.optionsToolBar)

        self.load_experiments()

        self.resultsWidget = QDockWidget("Experiments", self)
        self.experimentTable = ExperimentsModel(
            self.experiment_results, self.displayAsDistributionCheckbox, self
        )
        self.tableView = QTableView()
        self.tableView.setModel(self.experimentTable)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.selectionModel().selectionChanged.connect(
            self.onExperimentsSelectionChanged
        )
        for i in range(self.tableView.horizontalHeader().count()):
            self.tableView.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self.resultsWidget.setWidget(self.tableView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.resultsWidget)

        self.graphLabelsWidget = QDockWidget("Graph Labels", self)
        self.labelTable = GraphLabels(self.experiment_names, self)
        self.labelView = QTableView()
        self.labelView.setModel(self.labelTable)
        for i in range(self.labelView.horizontalHeader().count()):
            self.labelView.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.Stretch
            )
        self.graphLabelsWidget.setWidget(self.labelView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.graphLabelsWidget)

        self.labelTable.dataChanged.connect(self.onLabelChanged)
        self.displayAsDistributionCheckbox.stateChanged.connect(
            self.onChangeDisplayAsDistribution
        )

    def load_experiments(self):
        self.experiment_results = glob.glob(
            self.experiment_root + "**/{}".format(SOURCE), recursive=True
        )
        self.multirun_results = glob.glob(
            self.experiment_root + "**/{}".format(MULTI_ROOT), recursive=True
        )
        if self.displayAsDistributionCheckbox.checkState():
            for path in self.multirun_results:
                root = copy.deepcopy(path).replace("/" + MULTI_ROOT, "")

                self.experiment_results = list(
                    filter(lambda entry: root not in entry, self.experiment_results)
                )

                self.experiment_results.append(path)

        self.experiment_names = []
        for path in self.experiment_results:
            name = (
                path.replace(self.experiment_root, "")
                .replace("/{}".format(SOURCE), "")
                .replace("/{}".format(MULTI_ROOT), "")
            )
            self.experiment_names.append(name)

        global user_name_dict
        for name in self.experiment_names:
            if name not in user_name_dict:
                user_name_dict[name] = name

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

    def onLabelChanged(self, topLeft, bottomRight, roles):
        self.onExperimentsSelectionChanged()

    def onSaveFigure(self, checked):
        fname, _ = QFileDialog.getSaveFileName(
            self, caption="Save figure", filter="Figures (*.png *.pdf)"
        )
        if len(fname) > 0:
            self.chart.savefig(fname)

    def onChangeScale(self, state):
        self.onExperimentsSelectionChanged()

    def onChangeDisplayAsDistribution(self, state):
        self.load_experiments()
        self.experimentTable.reloadExperiments(self.experiment_results)

        self.onExperimentsSelectionChanged()

    def onExperimentsSelectionChanged(self, sel1=None, sel2=None):
        self.labelTable.reloadExperiments(self.experiment_names)

        selection = self.tableView.selectionModel()

        global user_name_dict
        if selection.hasSelection():
            self.axes.cla()

            for rowIndex in [row.row() for row in selection.selectedRows()]:
                if self.experiment_results[rowIndex].endswith(SOURCE):
                    result = pd.read_csv(self.experiment_results[rowIndex])

                    self.axes.plot(
                        result[XCOL],
                        result[YCOL],
                        label=self.experiment_names[rowIndex],
                    )
                elif self.experiment_results[rowIndex].endswith(MULTI_ROOT):
                    time_series = None
                    data_series = []
                    for path in glob.glob(
                        self.experiment_results[rowIndex].replace(MULTI_ROOT, "")
                        + "**/{}".format(SOURCE),
                        recursive=True,
                    ):
                        result = pd.read_csv(path)

                        time_series = (
                            result[XCOL] if time_series is None else time_series
                        )
                        data_series.append(result[YCOL])

                    df = pd.concat(data_series, axis=1)
                    mean_series = df.mean(axis=1)
                    var_series = df.std(axis=1)

                    mean_line = self.axes.plot(
                        time_series,
                        mean_series,
                        label=user_name_dict[self.experiment_names[rowIndex]],
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

            self.axes.set_xlabel(user_name_dict["x_label"])
            self.axes.set_ylabel(user_name_dict["y_label"])

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
