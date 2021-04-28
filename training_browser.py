from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import sys
import glob
import numpy as np
from scipy import stats
import pandas as pd


SOURCE = 'results.csv'
XCOL = 'env_step'
YCOL = 'episode_reward'
# SOURCE = 'model_train.csv'
# XCOL = 'step'
# YCOL = 'model_best_val_score'


class ExperimentsModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.data = [path.split('/') for path in data]
        self._headers = ['Algorithm', 'Sub-experiment', 'Environment', 'Date', 'Time']

    def rowCount(self, parent=None):
        return len(self.data)

    def columnCount(self, parent=None):
        return len(self.data[0])

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
    def __init__(self):
        super(BasicTrainingResultsWindow, self).__init__()

        self.experiment_root = QDir.currentPath() + '/exp/'
        self.experiment_results = glob.glob(self.experiment_root + '**/{}'.format(SOURCE), recursive=True)
        self.experiment_names = []
        for path in self.experiment_results:
            name = path.replace(self.experiment_root, '').replace('/{}'.format(SOURCE), '')
            name = name[:-4] + ':' + name[-4:-2] + ':' + name[-2:]
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

        self.optionsToolBar = QToolBar(self)
        self.optionsToolBar.addWidget(self.logYAxisCheckbox)
        self.optionsToolBar.addWidget(self.filterOutliersCheckbox)
        self.addToolBar(self.optionsToolBar)

        self.resultsWidget = QDockWidget('Experiments', self)
        self.experimentTable = ExperimentsModel(self.experiment_names, self)
        self.tableView = QTableView()
        self.tableView.setModel(self.experimentTable)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.selectionModel().selectionChanged.connect(self.onExperimentsSelectionChanged)
        self.resultsWidget.setWidget(self.tableView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.resultsWidget)

    def onChangeScale(self, state):
        self.onExperimentsSelectionChanged()

    def onFilterOutliers(self, state):
        self.onExperimentsSelectionChanged()

    def onExperimentsSelectionChanged(self, sel1=None, sel2=None):
        selection = self.tableView.selectionModel()

        if selection.hasSelection():
            self.chart.removeAllSeries()
            for axis in self.chart.axes():
                self.chart.removeAxis(axis)

            minX = None
            maxX = None
            minY = None
            maxY = None
            for rowIndex in [row.row() for row in selection.selectedRows()]:
                series = QLineSeries()
                series.setName(self.experiment_names[rowIndex])

                result = pd.read_csv(self.experiment_results[rowIndex])
                if self.filterOutliersCheckbox.checkState():
                    result = result[(np.abs(stats.zscore(result[YCOL])) < 3)]

                minX = result[XCOL].min() if minX is None else min(minX, result[XCOL].min())
                maxX = result[XCOL].max() if maxX is None else max(maxX, result[XCOL].max())
                minY = result[YCOL].min() if minY is None else min(minY, result[YCOL].min())
                maxY = result[YCOL].max() if maxY is None else max(maxY, result[YCOL].max())
                for x, y in zip(result[XCOL], result[YCOL]):
                    series.append(x, y)
                self.chart.addSeries(series)

            xAxis = QValueAxis()
            xAxis.setMin(minX)
            xAxis.setMax(maxX)
            yAxis = QLogValueAxis() if self.logYAxisCheckbox.checkState() else QValueAxis()
            yAxis.setMin(minY)
            yAxis.setMax(maxY)
            self.chart.addAxis(xAxis, Qt.AlignBottom)
            self.chart.addAxis(yAxis, Qt.AlignLeft)
            for series in self.chart.series():
                series.attachAxis(xAxis)
                series.attachAxis(yAxis)
                

if __name__ == '__main__':
    # create the application and the main window
    app = QApplication(sys.argv)
    window = BasicTrainingResultsWindow()

    # run
    window.showMaximized()
    app.exec_()
