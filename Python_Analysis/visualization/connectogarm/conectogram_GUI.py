
from threading import *
import wx
import sys
sys.path.append('./funcs/')
import time
import numpy as np
import os
import pandas as pd
import wx.lib.sized_controls as sc
import tkinter as tk
from tkinter import filedialog
import read_data as rd
from threading import *
import wx
import sys

sys.path.append('./funcs/')

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.path import Path
from more_itertools import unique_everseen
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import itertools
import seaborn as sns

import read_data as rd
import plot_funcs as pf
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
import plot_connectogram

sub_path  ='X:\\4 e-Lab\\' # y:\\eLab


print('start')
import wx
import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class TopPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent)

        self.figure = Figure() #figsize=(30,30)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(self.sizer)
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("A/D Counts")

    def draw(self,plot_main,data_edges):
        plot_main.plot_con(data_edges,self.axes)

    def clear(self):
        x = np.arange(0, 3, 0.01)
        y = np.sin(np.pi * x)
        #self.axes.plot(x, y)
        self.axes.cla()


class BottomPanel(wx.Panel):
    def __init__(self, parent, top):
        wx.Panel.__init__(self, parent=parent)

        self.graph = top

        self.togglebuttonStart = wx.ToggleButton(self, id=-1, label="Clear", pos=(10, 10))
        self.togglebuttonStart.Bind(wx.EVT_TOGGLEBUTTON, self.OnStartClick)

        labelChannels = wx.StaticText(self, -1, "Analog Inputs", pos=(200, 10))
        self.cb1 = wx.CheckBox(self, -1, label="A0", pos=(200, 30))
        self.cb2 = wx.CheckBox(self, -1, label="A1", pos=(200, 45))
        self.cb3 = wx.CheckBox(self, -1, label="A2", pos=(200, 60))
        self.cb4 = wx.CheckBox(self, -1, label="A3", pos=(200, 75))
        self.Bind(wx.EVT_CHECKBOX, self.OnChecked)

        self.textboxSampleTime = wx.TextCtrl(self, -1, "1000", pos=(200, 115), size=(50, -1))
        self.buttonSend = wx.Button(self, -1, "Send", pos=(250, 115), size=(50, -1))
        self.buttonSend.Bind(wx.EVT_BUTTON, self.OnSend)

        labelMinY = wx.StaticText(self, -1, "Min Y ", pos=(400, 10))
        self.textboxMinYAxis = wx.TextCtrl(self, -1, "0", pos=(400, 30))
        labelMaxY = wx.StaticText(self, -1, "Max Y", pos=(400, 60))
        self.textboxMaxYAxis = wx.TextCtrl(self, -1, "1024", pos=(400, 80))

        self.buttonRange = wx.Button(self, -1, "Set Y Axis", pos=(400, 105))
        self.buttonRange.Bind(wx.EVT_BUTTON, self.SetButtonRange)

        ## 1. get data
        data_con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con_all.csv'
        data_con_all = pd.read_csv(data_con_file)
        data_con = data_con_all[~np.isnan(data_con_all.Dir_index)]
        data_con = data_con.reset_index(drop=True)
        chan_ID = np.unique(np.concatenate([data_con.Stim, data_con.Chan])).astype('int')
        data_nodes = rd.get_nodes(chan_ID, data_con)
        #
        ### Load plot information
        self.plot_main = plot_connectogram.main_plot(data_con, data_nodes)

        # edges that will be modified
        data_edges = data_con[(data_con.d > 80) & (data_con.d < 100) & (data_con.Dir_index == 1)]
        self.data_edges = data_edges.reset_index(drop=True)

    def SetButtonRange(self, event):
        min = self.textboxMinYAxis.GetValue()
        max = self.textboxMaxYAxis.GetValue()
        #self.graph.changeAxes(min, max)

    def OnSend(self, event):
        val = self.textboxSampleTime.GetValue()
        print(val)
        self.graph.draw(self.plot_main, self.data_edges)
        # self.plot_main.plot_con(self.data_edges)

    def OnChecked(self, event):
        cb = event.GetEventObject()
        print("%s is clicked" % (cb.GetLabel()))

    def OnStartClick(self, event):
        self.graph.clear()


class Main(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title="Arduino Oscilloscope", size=(600, 600))

        splitter = wx.SplitterWindow(self)
        top = TopPanel(splitter)
        bottom = BottomPanel(splitter, top)
        splitter.SplitHorizontally(top, bottom)
        splitter.SetMinimumPaneSize(400)
        #top.draw()


if __name__ == "__main__":
    app = wx.App()
    frame = Main()
    frame.Show()
    app.MainLoop()
#
# class WorkerThread_Clinic(Thread):
#     """Worker Thread Class."""
#
#     # This is the function that will run the stimulation protocol for clinic stimulations.
#     # it is a different thread for simplicity reasons.
#     # in clinic use, the stimulation parameters may change (widht, frequency duration etc)
#     def __init__(self, notify_window, table, stim_num, labels, block_num, com):
#         """Init Worker Thread Class."""
#         Thread.__init__(self)
#         self._notify_window = notify_window
#         self._want_abort = 0
#         # This starts the thread running on creation, but you could
#         # also make the GUI thread responsible for calling this
#         # self.par            = par
#         self.stim_num = stim_num
#         self.all_labels = labels
#         self.block_num = block_num
#         self.block_max = block_num + 1
#         self.log = logfile.main()
#         self.table = table
#         self.start()
#         self.com = com
#
#     def run(self):
#         i = 0
#         protocol = 'Clinic'
#         while self._want_abort == 0:  # repeat for number of blocks
#             table = np.array(self.table.values)
#
#             if self._want_abort:
#                 break
#
#             chan = [table[i, 0], table[i, 1]]
#             # print('Opening Channels: ', chan[0], chan[1], '----', self.all_labels[np.int64(table[i, 0] - 1)], '_',
#             #      self.all_labels[np.int64(table[i, 1] - 1)])
#             print('Opening Channels: ', chan[0], chan[1])
#             SM.open_chan(N_stim=self.stim_num, channels=chan, log=False)
#             w = 0.2
#             if table[i, 4] == 0:
#
#                 print(self.stim_num, '. SP Stimulation - Int:  ', table[i, 2], 'mA, w:', table[i, 5], 'ms (total)')
#                 w = stim.Stim_Clinic(chan=chan, Int_p=table[i, 2], num_pulse=1, f=2, w=table[i, 5] * 1000,
#                                      b=self.block_num, i=self.stim_num, logState=protocol, com=self.com)
#             else:
#                 print(self.stim_num, '. Stimulation - Int:  ', table[i, 2], 'mA, Fs:', table[i, 3], 'Hz, dur: ',
#                       table[i, 4],
#                       's, #Pulses: ', np.int64(table[i, 3] * table[i, 4]), ' w:', table[i, 5], 'ms (total)')
#                 w = stim.Stim_Clinic(chan=chan, Int_p=table[i, 2], num_pulse=np.int64(table[i, 3] * table[i, 4]),
#                                      f=table[i, 3], w=table[i, 5] * 1000, b=self.block_num, i=self.stim_num,
#                                      logState=protocol, com=self.com)
#             # par = [0, table[i, 2], 0, 0, 0, 0, table[i, 3], table[i, 5] * 1000, table[i, 4]]
#             # self.log.stim_clinic(state=protocol, channel=chan, parameter=par, num_stim=self.stim_num)
#
#             time.sleep(0.3 + table[i, 4])
#             # print('close')
#             # todo: add SM
#             SM.close_chan(channels=chan, log=False)
#             SM.reset()
#             # wait at least 0.3s before starting next stimulation
#             self.stim_num = self.stim_num + 1
#             self._want_abort = 1
#
#     def abort(self):
#         self._want_abort = 1
# import sys
# from PyQt5 import QtWidgets, QtCore
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# import matplotlib.pyplot as plt
#
#
# class PlotViewer(QtWidgets.QWidget):
#
#     doubleClickAction = QtCore.pyqtSignal(str)
#
#     def __init__(self, parent=None):
#         super(PlotViewer, self).__init__(parent)
#
#         self.figure = plt.figure(figsize=(5, 5))
#         self.figureCanvas = FigureCanvas(self.figure)
#         self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
#
#         # create main layout
#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(self.navigationToolbar)
#         layout.addWidget(self.figureCanvas)
#         self.setLayout(layout)
#
#         # create an axis
#         x = range(0, 10)
#         y = range(0, 20, 2)
#         ax = self.figure.add_subplot(111)
#         ax.plot(x, y)
#
#         '
#         if os.path.exists(data_con_file):### 1. get data
# #         data_con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con_all.csv
#             data_con_all = pd.read_csv(data_con_file)
#         data_con = data_con_all[~np.isnan(data_con_all.Dir_index)]
#         data_con = data_con.reset_index(drop=True)
#         chan_ID = np.unique(np.concatenate([data_con.Stim, data_con.Chan])).astype('int')
#         data_nodes = rd.get_nodes(chan_ID, data_con)
#         #
#         ### Load plot information
#         self.plot_main = plot_connectogram.main_plot(data_con, data_nodes)
#
#         # edges that will be modified
#         data_edges = data_con[(data_con.d > 80) & (data_con.d < 100) & (data_con.Dir_index == 1)]
#         self.data_edges = data_edges.reset_index(drop=True)
#
#
#         # show canvas
#         self.figureCanvas.show()
#
#     def create_plot(self, event):
#         self.plot_main.plot_con(self.data_edges)
#
#     def rem_plot(self, event):
#         self.plot_main.plot_con(self.data_edges)
#
#
# if __name__ == "__main__":
#
#     app = QtWidgets.QApplication(sys.argv)
#     widget = PlotViewer()
#     widget.show()
#     app.exec_()
#
#
#
