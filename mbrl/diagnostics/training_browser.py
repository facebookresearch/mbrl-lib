from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import os
import sys
import glob
import numpy as np
from scipy import stats
import pandas as pd
from argparse import ArgumentParser
import yaml
import signal


SOURCE = "results.csv"
XCOL = "env_step"
YCOL = "episode_reward"
# SOURCE = 'model_train.csv'
# XCOL = 'step'
# YCOL = 'model_best_val_score'
USE_AREA_DISTRIBUTION = False


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

        if experiment_root[-1] != "/":
            experiment_root = experiment_root + "/"
        self.experiment_results = glob.glob(
            experiment_root + "**/{}".format(SOURCE), recursive=True
        )
        self.experiment_names = []
        for path in self.experiment_results:
            name = path.replace(experiment_root, "").replace("/{}".format(SOURCE), "")
            self.experiment_names.append(name)

        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.AllAnimations)

        self.graphWidget = QChartView(self.chart)
        self.graphWidget.setRenderHint(QPainter.Antialiasing)
        self.setCentralWidget(self.graphWidget)

        self.logYAxisCheckbox = QCheckBox("Log Scale (Y Axis)")
        self.logYAxisCheckbox.stateChanged.connect(self.onChangeScale)

        self.filterOutliersCheckbox = QCheckBox("Filter Outliers")
        self.filterOutliersCheckbox.stateChanged.connect(self.onFilterOutliers)

        self.displayAsDistributionCheckbox = QCheckBox("Display As Distribution")
        self.displayAsDistributionCheckbox.stateChanged.connect(
            self.onChangeDisplayAsDistribution
        )
        self.displayAsDistributionCheckbox.setEnabled(False)

        self.optionsToolBar = QToolBar(self)
        self.optionsToolBar.addWidget(self.logYAxisCheckbox)
        self.optionsToolBar.addWidget(self.filterOutliersCheckbox)
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

    def onChangeScale(self, state):
        self.onExperimentsSelectionChanged()

    def onFilterOutliers(self, state):
        self.onExperimentsSelectionChanged()

    def onChangeDisplayAsDistribution(self, state):
        self.onExperimentsSelectionChanged()

    def onExperimentsSelectionChanged(self, sel1=None, sel2=None):
        selection = self.tableView.selectionModel()

        if selection.hasSelection():
            self.chart.removeAllSeries()
            for axis in self.chart.axes():
                self.chart.removeAxis(axis)

            self.displayAsDistributionCheckbox.setEnabled(
                self.selectedMatchingSequences()
            )
            displayAsDistribution = (
                self.displayAsDistributionCheckbox.isEnabled()
                and self.displayAsDistributionCheckbox.checkState()
            )

            minX = None
            maxX = None
            minY = None
            maxY = None
            series = []
            for rowIndex in [row.row() for row in selection.selectedRows()]:
                line_series = QLineSeries()
                line_series.setName(self.experiment_names[rowIndex])

                result = pd.read_csv(self.experiment_results[rowIndex])

                clipped = (
                    result[(np.abs(stats.zscore(result[YCOL])) < 3)]
                    if self.filterOutliersCheckbox.checkState()
                    else result
                )

                minX = (
                    clipped[XCOL].min()
                    if minX is None
                    else min(minX, clipped[XCOL].min())
                )
                maxX = (
                    clipped[XCOL].max()
                    if maxX is None
                    else max(maxX, clipped[XCOL].max())
                )
                minY = (
                    clipped[YCOL].min()
                    if minY is None
                    else min(minY, clipped[YCOL].min())
                )
                maxY = (
                    clipped[YCOL].max()
                    if maxY is None
                    else max(maxY, clipped[YCOL].max())
                )

                for x, y in zip(result[XCOL], result[YCOL]):
                    line_series.append(x, y)

                if displayAsDistribution:
                    if len(series) == 0:
                        series.append(result[XCOL])
                    series.append(result[YCOL])
                else:
                    self.chart.addSeries(line_series)

            if displayAsDistribution:
                time_series = series[0]
                data_series = series[1:]

                df = pd.concat(data_series, axis=1)
                mean_series = df.mean(axis=1)
                var_series = df.std(axis=1)

                line_series = QLineSeries()
                line_series.setName(self.experiment_names[rowIndex])
                for x, y in zip(time_series, mean_series):
                    line_series.append(x, y)
                self.chart.addSeries(line_series)

                dist_color = line_series.color()
                dist_color.setHsvF(
                    dist_color.hueF(),
                    0.5 * dist_color.saturationF(),
                    dist_color.valueF(),
                )

                lower_series = QLineSeries()
                for x, y in zip(time_series, mean_series.sub(other=var_series)):
                    lower_series.append(x, y)
                if not USE_AREA_DISTRIBUTION:
                    lower_series.setName("Minus one standard deviation")
                    lower_series.setColor(dist_color)
                    self.chart.addSeries(lower_series)

                upper_series = QLineSeries()
                for x, y in zip(time_series, mean_series.add(other=var_series)):
                    upper_series.append(x, y)
                if not USE_AREA_DISTRIBUTION:
                    upper_series.setName("Plus one standard deviation")
                    upper_series.setPen(lower_series.pen())
                    self.chart.addSeries(upper_series)

                if USE_AREA_DISTRIBUTION:
                    area_series = QAreaSeries(lower_series, upper_series)
                    area_series.setColor(dist_color)
                    self.chart.addSeries(area_series)

            xAxis = QValueAxis()
            xAxis.setMin(minX)
            xAxis.setMax(maxX)

            use_log = (minY > 0.0) and self.logYAxisCheckbox.checkState()
            yAxis = QLogValueAxis() if use_log else QValueAxis()
            yAxis.setMin(minY)
            yAxis.setMax(maxY)
            self.chart.addAxis(xAxis, Qt.AlignBottom)
            self.chart.addAxis(yAxis, Qt.AlignLeft)
            for series in self.chart.series():
                series.attachAxis(xAxis)
                series.attachAxis(yAxis)


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
