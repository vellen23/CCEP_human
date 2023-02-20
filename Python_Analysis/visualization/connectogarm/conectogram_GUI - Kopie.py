
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


class WorkerThread_Clinic(Thread):
    """Worker Thread Class."""

    # This is the function that will run the stimulation protocol for clinic stimulations.
    # it is a different thread for simplicity reasons.
    # in clinic use, the stimulation parameters may change (widht, frequency duration etc)
    def __init__(self, notify_window, table, stim_num, labels, block_num, com):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self._notify_window = notify_window
        self._want_abort = 0
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        # self.par            = par
        self.stim_num = stim_num
        self.all_labels = labels
        self.block_num = block_num
        self.block_max = block_num + 1
        self.log = logfile.main()
        self.table = table
        self.start()
        self.com = com

    def run(self):
        i = 0
        protocol = 'Clinic'
        while self._want_abort == 0:  # repeat for number of blocks
            table = np.array(self.table.values)

            if self._want_abort:
                break

            chan = [table[i, 0], table[i, 1]]
            # print('Opening Channels: ', chan[0], chan[1], '----', self.all_labels[np.int64(table[i, 0] - 1)], '_',
            #      self.all_labels[np.int64(table[i, 1] - 1)])
            print('Opening Channels: ', chan[0], chan[1])
            SM.open_chan(N_stim=self.stim_num, channels=chan, log=False)
            w = 0.2
            if table[i, 4] == 0:

                print(self.stim_num, '. SP Stimulation - Int:  ', table[i, 2], 'mA, w:', table[i, 5], 'ms (total)')
                w = stim.Stim_Clinic(chan=chan, Int_p=table[i, 2], num_pulse=1, f=2, w=table[i, 5] * 1000,
                                     b=self.block_num, i=self.stim_num, logState=protocol, com=self.com)
            else:
                print(self.stim_num, '. Stimulation - Int:  ', table[i, 2], 'mA, Fs:', table[i, 3], 'Hz, dur: ',
                      table[i, 4],
                      's, #Pulses: ', np.int64(table[i, 3] * table[i, 4]), ' w:', table[i, 5], 'ms (total)')
                w = stim.Stim_Clinic(chan=chan, Int_p=table[i, 2], num_pulse=np.int64(table[i, 3] * table[i, 4]),
                                     f=table[i, 3], w=table[i, 5] * 1000, b=self.block_num, i=self.stim_num,
                                     logState=protocol, com=self.com)
            # par = [0, table[i, 2], 0, 0, 0, 0, table[i, 3], table[i, 5] * 1000, table[i, 4]]
            # self.log.stim_clinic(state=protocol, channel=chan, parameter=par, num_stim=self.stim_num)

            time.sleep(0.3 + table[i, 4])
            # print('close')
            # todo: add SM
            SM.close_chan(channels=chan, log=False)
            SM.reset()
            # wait at least 0.3s before starting next stimulation
            self.stim_num = self.stim_num + 1
            self._want_abort = 1

    def abort(self):
        self._want_abort = 1


# GUI Frame class that spins off the worker thread
class MainFrame(wx.Frame):
    """Class MainFrame."""

    def __init__(self, parent, id):
        # GUI structure with buttons
        super(MainFrame, self).__init__(parent, title='NELS', size=(1700, 1000))
        ### 1. get data
        data_con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con_all.csv'
        if os.path.exists(data_con_file):
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




        # 3. creating GUI with all the buttons and assign a function to the buttonss
        """Create the MainFrame."""
        self.SetBackgroundColour(wx.Colour(189, 215, 238))  # e-lab colour as background, bright blue
        menuBar = wx.MenuBar()
        #menu = wx.Menu()
        #menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
        # editButton = wx.Menu()
        #exitItem = menu.Append(wx.ID_EXIT, 'Exit', "Exit demo")  # , 'status msg'
        # self.Bind(wx.EVT_MENU, self.Quit, exitItem)
        #menuBar.Append(menu, '&File')  # & shortcut possbile
        # menuBar.Append(editButton, '&Edit')

        #self.SetMenuBar(menuBar)
        #self.Bind(wx.EVT_CHAR_HOOK, self.KeyPress)

        self.SetTitle('e-Lab')

        text = wx.StaticText(self, label="Neuro e-lab Connectogram", pos=(165, 10), size=(200, 40))
        text.SetFont(wx.Font(30, wx.DECORATIVE, wx.NORMAL, wx.NORMAL, False, u'Garamond'))
        font_log = wx.Font(18, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Garamond')
        font_but = wx.Font(15, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Garamond')
        font_small = wx.Font(12, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Garamond')

        x_r = 60
        y_r = 80

        button_l = 240
        button_h = 60
        #wx.StaticLine(self, wx.NewIdRef(count=1), pos=(2 * x_r + 2.2 * button_l, y_r), size=(1, 800), name='line')
        # button_Imp = wx.Button(self, wx.NewIdRef(count=1), 'Impedance Measurement', pos=(x_r, y_r + 2.4 * button_h),
        #                        size=(button_l, button_h))
        # button_Imp.Bind(wx.EVT_BUTTON, self._Imp)
        # button_Imp.SetFont(font_but)
        button_reload = wx.Button(self, wx.NewIdRef(count=1), 'Start', pos=(x_r, y_r + 2.4 * button_h),
                                  size=(button_l, button_h))
        button_reload.Bind(wx.EVT_BUTTON, self.create_plot)
        button_reload.SetFont(font_but)

        button_SM = wx.Button(self, wx.NewIdRef(count=1), 'Clear', pos=(x_r, y_r + 1.2 * button_h),
                              size=(button_l, button_h))  # wx.NewId()
        button_SM.Bind(wx.EVT_BUTTON, self.rem_plot)
        button_SM.SetFont(font_but)


        x_c = 3 * x_r + 2.2 * button_l
        y_c = 80

        self.log_control = wx.TextCtrl(self, wx.NewIdRef(count=1),
                                       style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_RICH,
                                       pos=(x_c, y_c + 180), size=(530, 585))
        self.log_control.SetFont(font_log)
        self.log_control.SetDefaultStyle(wx.TextAttr(wx.RED))
        # self.log_control.AppendText('--- Is neuralynx recording? ---\n')
        self.log_control.SetDefaultStyle(wx.TextAttr(wx.BLACK))

    def create_plot(self, event):
        self.plot_main.plot_con(self.data_edges)

    def rem_plot(self, event):
        self.plot_main.plot_con(self.data_edges)



#####Main script to run GUI
class MainApp(wx.App):
    """Class Main App."""

    def OnInit(self):
        """Init Main App."""
        self.frame = MainFrame(None, -1)
        self.frame.Show(True)
        self.SetTopWindow(self.frame)
        return True


if __name__ == '__main__':
    app = MainApp(0)
    app.MainLoop()
