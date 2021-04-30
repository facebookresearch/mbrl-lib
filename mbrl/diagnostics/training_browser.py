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
        self.data = []
        for path in [path.replace('/{}'.format(SOURCE), '/.hydra/config.yaml') for path in data]:
            config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
            entry = [config['algorithm']['name']]
            entry = entry + [config['experiment']]
            entry = entry + [config['env'] if 'env' in config else config['overrides']['env']]
            entry = entry + [config['seed']]
            self.data.append(entry)
        self._headers = ['Algorithm', 'Experiment', 'Environment', 'Seed']

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

        if experiment_root[-1] != '/':
            experiment_root = experiment_root + '/'
        self.experiment_results = glob.glob(experiment_root + '**/{}'.format(SOURCE), recursive=True)
        self.experiment_names = []
        for path in self.experiment_results:
            name = path.replace(experiment_root, '').replace('/{}'.format(SOURCE), '')
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
        self.experimentTable = ExperimentsModel(self.experiment_results, self)
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

            use_log = (minY > 0.0) and self.logYAxisCheckbox.checkState()
            yAxis = QLogValueAxis() if use_log else QValueAxis()
            yAxis.setMin(minY)
            yAxis.setMax(maxY)
            self.chart.addAxis(xAxis, Qt.AlignBottom)
            self.chart.addAxis(yAxis, Qt.AlignLeft)
            for series in self.chart.series():
                series.attachAxis(xAxis)
                series.attachAxis(yAxis)
                

if __name__ == '__main__':
    argparser = ArgumentParser(description='Visualize training results')
    argparser.add_argument('--experiments', type=str, default=QDir.currentPath() + '/exp/',
                        help='The path to the experiments folder')
    args = argparser.parse_args()

    if not os.path.exists(args.experiments):
        print('Path ' + args.experiments + ' does not exist.')
        exit(-1)

    # create the application and the main window
    app = QApplication(sys.argv)
    window = BasicTrainingResultsWindow(args.experiments)

    # run
    window.showMaximized()
    app.exec_()
