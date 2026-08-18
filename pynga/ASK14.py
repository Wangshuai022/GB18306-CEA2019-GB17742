#!/usr/bin/env python
from __future__ import division, print_function

import os
import numpy as np

from pynga.utils import (
    GetKey,
    calc_W,
    calc_dip,
    calc_Zhypo,
    calc_Ztor,
    mapfunc,
    calc_Rx,
    calc_Rrup,
    calc_Z1,
    rake2ftype_AS,
)


class ASK14_nga(object):
    """
    ASK14 NGA model class
    """

    def __init__(self):

        self.filepth = os.path.join(os.path.dirname(os.path.realpath(__file__)), "NGA_west2")
        self.CoefFile = os.path.join(self.filepth, "ASK14.csv")
        self.Coefs = {}
        self.read_model_coefs()
        self.countries = ["California", "Japan"]

        # region for regional corrections (Vs30 and Rrup)
        self.regions = [
            "CA",
            "TW",
            "CN",
            "JP",
        ]

        # New period independent parameters:
        self.M2 = 5
        self.a4 = -0.1
        self.a5 = -0.49
        self.a7 = 0
        self.a2HW = 0.2
        self.n = 1.5
        self.a39 = 0.0

    def read_model_coefs(self):
        # self.CoefKeys = open(self.CoefFile, "r").readlines()[1].strip().split(",")[1:]
        # inputs = np.loadtxt(self.CoefFile, skiprows=2, delimiter=",")

        # self.periods = inputs[:, 0]
        # coefs = inputs[:, 1:]
        # for i in range(len(self.periods)):
        #     T1 = self.periods[i]
        #     Tkey = GetKey(T1)

        #     # periods list ( -2: PGV, -1: PGA ) (mapping between the
        #     # NGA models accordingly, -1: PGV, 0: PGA)
        #     if Tkey == "-1.000":
        #         Tkey = "-2.000"  # PGV
        #         self.periods[i] = -2
        #     if Tkey == "0.000":
        #         Tkey = "-1.000"  # PGA
        #         self.periods[i] = -1

        #     self.Coefs[Tkey] = {}
        #     for ikey in range(len(self.CoefKeys)):
        #         key = self.CoefKeys[ikey]
        #         cmd = "self.Coefs['%s']['%s'] = coefs[%i,%i]" % (Tkey, key, i, ikey)
        #         exec(cmd)

        self.CoefKeys = [
            "VLIN",
            "b",
            "c4",
            "M1",
            "a1",
            "a2",
            "a3",
            "a6",
            "a8",
            "a10",
            "a11",
            "a12",
            "a13",
            "a14",
            "a15",
            "a17",
            "a43",
            "a44",
            "a45",
            "a46",
            "a25",
            "a28",
            "a29",
            "a31",
            "a36",
            "a37",
            "a38",
            "a40",
            "a41",
            "a42",
            "s01",
            "s02",
            "s11",
            "s12",
            "s3",
            "s4",
            "s5",
            "s6",
        ]
        inputs = np.array(
            [
                [
                    0.0000,
                    660.0000,
                    -1.4700,
                    6.0000,
                    6.7500,
                    0.4640,
                    -0.7900,
                    0.2810,
                    2.2800,
                    0.0000,
                    1.7350,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0066,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0015,
                    0.0025,
                    -0.0034,
                    -0.1503,
                    0.2650,
                    0.3370,
                    0.1880,
                    0.0880,
                    -0.1960,
                    0.0440,
                    0.7540,
                    0.5200,
                    0.7410,
                    0.5010,
                    0.4700,
                    0.3600,
                    0.5400,
                    0.6300,
                ],
                [
                    -1.0000,
                    330.0000,
                    -2.0200,
                    3.0000,
                    6.7500,
                    6.1680,
                    -0.9500,
                    0.2810,
                    2.3000,
                    -0.1200,
                    2.3600,
                    0.0000,
                    -0.1000,
                    0.2500,
                    0.2200,
                    0.9000,
                    -0.0010,
                    0.2800,
                    0.1500,
                    0.0900,
                    0.0700,
                    -0.0001,
                    0.0005,
                    -0.0037,
                    -0.1462,
                    0.3770,
                    0.2120,
                    0.1570,
                    0.0950,
                    -0.0380,
                    0.0650,
                    0.6620,
                    0.5100,
                    0.6600,
                    0.5100,
                    0.3800,
                    0.3800,
                    0.5800,
                    0.5300,
                ],
                [
                    0.0100,
                    660.0000,
                    -1.4700,
                    6.0000,
                    6.7500,
                    0.4640,
                    -0.7900,
                    0.2810,
                    2.2800,
                    0.0000,
                    1.7350,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0066,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0015,
                    0.0025,
                    -0.0034,
                    -0.1503,
                    0.2650,
                    0.3370,
                    0.1880,
                    0.0880,
                    -0.1960,
                    0.0440,
                    0.7540,
                    0.5200,
                    0.7410,
                    0.5010,
                    0.4700,
                    0.3600,
                    0.5400,
                    0.6300,
                ],
                [
                    0.0200,
                    680.0000,
                    -1.4600,
                    6.0000,
                    6.7500,
                    0.4730,
                    -0.7900,
                    0.2810,
                    2.2800,
                    0.0000,
                    1.7180,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0066,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0015,
                    0.0024,
                    -0.0033,
                    -0.1479,
                    0.2550,
                    0.3280,
                    0.1840,
                    0.0880,
                    -0.1940,
                    0.0610,
                    0.7600,
                    0.5200,
                    0.7470,
                    0.5010,
                    0.4700,
                    0.3600,
                    0.5400,
                    0.6300,
                ],
                [
                    0.0300,
                    770.0000,
                    -1.3900,
                    6.0000,
                    6.7500,
                    0.4570,
                    -0.7900,
                    0.2810,
                    2.2500,
                    0.0000,
                    1.6150,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0066,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0016,
                    0.0023,
                    -0.0034,
                    -0.1447,
                    0.2490,
                    0.3200,
                    0.1800,
                    0.0930,
                    -0.1750,
                    0.1620,
                    0.7810,
                    0.5200,
                    0.7690,
                    0.5010,
                    0.4700,
                    0.3600,
                    0.5500,
                    0.6300,
                ],
                [
                    0.0500,
                    800.0000,
                    -1.2200,
                    6.0000,
                    6.7500,
                    0.6520,
                    -0.7900,
                    0.2810,
                    2.1800,
                    0.0000,
                    1.3580,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0075,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0020,
                    0.0027,
                    -0.0033,
                    -0.1326,
                    0.2020,
                    0.2890,
                    0.1670,
                    0.1330,
                    -0.0900,
                    0.4510,
                    0.8100,
                    0.5300,
                    0.7980,
                    0.5120,
                    0.4700,
                    0.3600,
                    0.5600,
                    0.6500,
                ],
                [
                    0.0750,
                    800.0000,
                    -1.1500,
                    6.0000,
                    6.7500,
                    0.9500,
                    -0.7900,
                    0.2780,
                    2.1300,
                    0.0000,
                    1.2580,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0092,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0027,
                    0.0032,
                    -0.0029,
                    -0.1353,
                    0.1260,
                    0.2750,
                    0.1730,
                    0.1860,
                    0.0900,
                    0.5060,
                    0.8100,
                    0.5400,
                    0.7980,
                    0.5220,
                    0.4700,
                    0.3600,
                    0.5700,
                    0.6900,
                ],
                [
                    0.1000,
                    800.0000,
                    -1.2300,
                    5.9000,
                    6.7500,
                    1.1600,
                    -0.7900,
                    0.2700,
                    2.1400,
                    0.0000,
                    1.3100,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0101,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0033,
                    0.0036,
                    -0.0025,
                    -0.1128,
                    0.0220,
                    0.2560,
                    0.1890,
                    0.1600,
                    0.0060,
                    0.3350,
                    0.8100,
                    0.5500,
                    0.7950,
                    0.5270,
                    0.4700,
                    0.3600,
                    0.5700,
                    0.7000,
                ],
                [
                    0.1500,
                    740.0000,
                    -1.5900,
                    5.8000,
                    6.7500,
                    1.4870,
                    -0.7900,
                    0.2580,
                    2.1900,
                    -0.0290,
                    1.6600,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0097,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0500,
                    -0.0035,
                    0.0033,
                    -0.0025,
                    0.0383,
                    -0.1360,
                    0.1620,
                    0.1080,
                    0.0680,
                    -0.1560,
                    -0.0840,
                    0.8010,
                    0.5600,
                    0.7730,
                    0.5190,
                    0.4700,
                    0.3600,
                    0.5800,
                    0.7000,
                ],
                [
                    0.2000,
                    590.0000,
                    -2.0100,
                    5.7000,
                    6.7500,
                    1.7120,
                    -0.7900,
                    0.2500,
                    2.2500,
                    -0.0500,
                    2.2200,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.3000,
                    1.1000,
                    -0.0084,
                    0.1000,
                    0.0500,
                    0.0000,
                    -0.0300,
                    -0.0033,
                    0.0027,
                    -0.0031,
                    0.0775,
                    -0.0780,
                    0.2240,
                    0.1150,
                    0.0480,
                    -0.2740,
                    -0.1780,
                    0.7890,
                    0.5650,
                    0.7530,
                    0.5140,
                    0.4700,
                    0.3600,
                    0.5900,
                    0.7000,
                ],
                [
                    0.2500,
                    495.0000,
                    -2.4100,
                    5.6000,
                    6.7500,
                    1.7960,
                    -0.7900,
                    0.2420,
                    2.3000,
                    -0.0660,
                    2.7700,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.2400,
                    1.1000,
                    -0.0074,
                    0.1000,
                    0.0500,
                    0.0000,
                    0.0000,
                    -0.0029,
                    0.0024,
                    -0.0036,
                    0.0741,
                    0.0370,
                    0.2480,
                    0.1220,
                    0.0550,
                    -0.2480,
                    -0.1870,
                    0.7700,
                    0.5700,
                    0.7290,
                    0.5130,
                    0.4700,
                    0.3600,
                    0.6100,
                    0.7000,
                ],
                [
                    0.3000,
                    430.0000,
                    -2.7600,
                    5.5000,
                    6.7500,
                    1.8490,
                    -0.7900,
                    0.2390,
                    2.3500,
                    -0.0790,
                    3.2500,
                    0.0000,
                    -0.1000,
                    0.6000,
                    -0.1900,
                    1.0300,
                    -0.0064,
                    0.1000,
                    0.0500,
                    0.0300,
                    0.0300,
                    -0.0027,
                    0.0020,
                    -0.0039,
                    0.2548,
                    -0.0910,
                    0.2030,
                    0.0960,
                    0.0730,
                    -0.2030,
                    -0.1590,
                    0.7400,
                    0.5800,
                    0.6930,
                    0.5190,
                    0.4700,
                    0.3600,
                    0.6300,
                    0.7000,
                ],
                [
                    0.4000,
                    360.0000,
                    -3.2800,
                    5.2000,
                    6.7500,
                    1.8250,
                    -0.7900,
                    0.2310,
                    2.4500,
                    -0.0990,
                    3.9900,
                    0.0000,
                    -0.1000,
                    0.5800,
                    -0.1100,
                    0.9200,
                    -0.0043,
                    0.1000,
                    0.0700,
                    0.0600,
                    0.0600,
                    -0.0023,
                    0.0010,
                    -0.0048,
                    0.2136,
                    0.1290,
                    0.2320,
                    0.1230,
                    0.1430,
                    -0.1540,
                    -0.0230,
                    0.6990,
                    0.5900,
                    0.6440,
                    0.5240,
                    0.4700,
                    0.3600,
                    0.6600,
                    0.7000,
                ],
                [
                    0.5000,
                    340.0000,
                    -3.6000,
                    4.8000,
                    6.7500,
                    1.7680,
                    -0.7900,
                    0.2300,
                    2.5500,
                    -0.1150,
                    4.4500,
                    0.0000,
                    -0.1000,
                    0.5600,
                    -0.0400,
                    0.8400,
                    -0.0032,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.0900,
                    -0.0020,
                    0.0008,
                    -0.0050,
                    0.1542,
                    0.3100,
                    0.2520,
                    0.1340,
                    0.1600,
                    -0.1590,
                    -0.0290,
                    0.6760,
                    0.6000,
                    0.6160,
                    0.5320,
                    0.4700,
                    0.3600,
                    0.6900,
                    0.7000,
                ],
                [
                    0.7500,
                    330.0000,
                    -3.8000,
                    4.4000,
                    6.7500,
                    1.5430,
                    -0.7900,
                    0.2300,
                    2.6500,
                    -0.1440,
                    4.7500,
                    0.0000,
                    -0.1000,
                    0.5300,
                    0.0700,
                    0.6800,
                    -0.0025,
                    0.1400,
                    0.1400,
                    0.1400,
                    0.1300,
                    -0.0010,
                    0.0007,
                    -0.0041,
                    0.0787,
                    0.5050,
                    0.2080,
                    0.1290,
                    0.1580,
                    -0.1410,
                    0.0610,
                    0.6310,
                    0.6150,
                    0.5660,
                    0.5480,
                    0.4700,
                    0.3600,
                    0.7300,
                    0.6900,
                ],
                [
                    1.0000,
                    330.0000,
                    -3.5000,
                    4.0000,
                    6.7500,
                    1.2920,
                    -0.7900,
                    0.2300,
                    2.7000,
                    -0.1650,
                    4.3000,
                    0.0000,
                    -0.1000,
                    0.5000,
                    0.1500,
                    0.5700,
                    -0.0022,
                    0.1700,
                    0.1700,
                    0.1700,
                    0.1400,
                    -0.0005,
                    0.0007,
                    -0.0032,
                    0.0476,
                    0.3580,
                    0.2080,
                    0.1520,
                    0.1450,
                    -0.1440,
                    0.0620,
                    0.6090,
                    0.6300,
                    0.5410,
                    0.5650,
                    0.4700,
                    0.3600,
                    0.7700,
                    0.6800,
                ],
                [
                    1.5000,
                    330.0000,
                    -2.4000,
                    3.7500,
                    6.7500,
                    0.8550,
                    -0.7900,
                    0.2300,
                    2.7500,
                    -0.1940,
                    2.6500,
                    0.0000,
                    -0.1000,
                    0.4200,
                    0.2700,
                    0.4200,
                    -0.0016,
                    0.2200,
                    0.2100,
                    0.2000,
                    0.1600,
                    -0.0004,
                    0.0006,
                    -0.0020,
                    -0.0163,
                    0.1310,
                    0.1080,
                    0.1180,
                    0.1310,
                    -0.1260,
                    0.0370,
                    0.5780,
                    0.6400,
                    0.5060,
                    0.5760,
                    0.4700,
                    0.3600,
                    0.8000,
                    0.6600,
                ],
                [
                    2.0000,
                    330.0000,
                    -1.0000,
                    3.5000,
                    6.7500,
                    0.5210,
                    -0.7900,
                    0.2300,
                    2.7500,
                    -0.2140,
                    0.5500,
                    0.0000,
                    -0.1000,
                    0.3500,
                    0.3500,
                    0.3100,
                    -0.0013,
                    0.2600,
                    0.2500,
                    0.2200,
                    0.1600,
                    -0.0002,
                    0.0003,
                    -0.0017,
                    -0.1203,
                    0.1230,
                    0.0680,
                    0.1190,
                    0.0830,
                    -0.0750,
                    -0.1430,
                    0.5550,
                    0.6500,
                    0.4800,
                    0.5870,
                    0.4700,
                    0.3600,
                    0.8000,
                    0.6200,
                ],
                [
                    3.0000,
                    330.0000,
                    0.0000,
                    3.2500,
                    6.8200,
                    0.1600,
                    -0.7900,
                    0.2300,
                    2.7500,
                    -0.2430,
                    -0.9500,
                    0.0000,
                    -0.1000,
                    0.2000,
                    0.4600,
                    0.1600,
                    -0.0010,
                    0.3400,
                    0.3000,
                    0.2300,
                    0.1600,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.2719,
                    0.1090,
                    -0.0230,
                    0.0930,
                    0.0700,
                    -0.0210,
                    -0.0280,
                    0.5480,
                    0.6400,
                    0.4720,
                    0.5760,
                    0.4700,
                    0.3600,
                    0.8000,
                    0.5500,
                ],
                [
                    4.0000,
                    330.0000,
                    0.0000,
                    3.0000,
                    6.9200,
                    -0.0700,
                    -0.7900,
                    0.2300,
                    2.7500,
                    -0.2640,
                    -0.9500,
                    0.0000,
                    -0.1000,
                    0.0000,
                    0.5400,
                    0.0500,
                    -0.0010,
                    0.4100,
                    0.3200,
                    0.2300,
                    0.1400,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.2958,
                    0.1350,
                    0.0280,
                    0.0840,
                    0.1010,
                    0.0720,
                    -0.0970,
                    0.5270,
                    0.6300,
                    0.4470,
                    0.5650,
                    0.4700,
                    0.3600,
                    0.7600,
                    0.5200,
                ],
                [
                    5.0000,
                    330.0000,
                    0.0000,
                    3.0000,
                    7.0000,
                    -0.4100,
                    -0.7560,
                    0.2300,
                    2.7500,
                    -0.2700,
                    -0.9300,
                    0.0000,
                    -0.1000,
                    0.0000,
                    0.6100,
                    -0.0400,
                    -0.0010,
                    0.5100,
                    0.3200,
                    0.2200,
                    0.1300,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.2718,
                    0.1890,
                    0.0310,
                    0.0580,
                    0.0950,
                    0.2050,
                    0.0150,
                    0.5050,
                    0.6300,
                    0.4250,
                    0.5680,
                    0.4700,
                    0.3600,
                    0.7200,
                    0.5000,
                ],
                [
                    6.0000,
                    330.0000,
                    0.0000,
                    3.0000,
                    7.0600,
                    -0.8380,
                    -0.7000,
                    0.2300,
                    2.7500,
                    -0.2700,
                    -0.9100,
                    0.0000,
                    -0.1000,
                    0.0000,
                    0.6500,
                    -0.1100,
                    -0.0010,
                    0.5500,
                    0.3200,
                    0.2000,
                    0.1000,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.2517,
                    0.2150,
                    0.0240,
                    0.0650,
                    0.1330,
                    0.2850,
                    0.1040,
                    0.4770,
                    0.6300,
                    0.3950,
                    0.5710,
                    0.4700,
                    0.3600,
                    0.7000,
                    0.5000,
                ],
                [
                    7.5000,
                    330.0000,
                    0.0000,
                    3.0000,
                    7.1500,
                    -1.4330,
                    -0.6200,
                    0.2300,
                    2.7500,
                    -0.2700,
                    -0.8750,
                    0.0000,
                    -0.1000,
                    0.0000,
                    0.7200,
                    -0.1900,
                    -0.0010,
                    0.5500,
                    0.2900,
                    0.1700,
                    0.0800,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.1337,
                    0.1660,
                    -0.0610,
                    0.0090,
                    0.1510,
                    0.3290,
                    0.2990,
                    0.4570,
                    0.6300,
                    0.3780,
                    0.5750,
                    0.4700,
                    0.3600,
                    0.6700,
                    0.5000,
                ],
                [
                    10.0000,
                    330.0000,
                    0.0000,
                    3.0000,
                    7.2500,
                    -2.3680,
                    -0.5150,
                    0.2300,
                    2.7500,
                    -0.2700,
                    -0.8000,
                    0.0000,
                    -0.1000,
                    0.0000,
                    0.8000,
                    -0.3000,
                    -0.0010,
                    0.4200,
                    0.2200,
                    0.1400,
                    0.0800,
                    0.0000,
                    0.0000,
                    -0.0020,
                    -0.0216,
                    0.0920,
                    -0.1590,
                    -0.0500,
                    0.1240,
                    0.3010,
                    0.2430,
                    0.4290,
                    0.6300,
                    0.3590,
                    0.5850,
                    0.4700,
                    0.3600,
                    0.6400,
                    0.5000,
                ],
            ],
            dtype=np.float32,
        )
        # 保留原始处理逻辑
        self.periods = inputs[:, 0]
        coefs = inputs[:, 1:]

        # 周期键映射（保持原逻辑）
        for i in range(len(self.periods)):
            T1 = self.periods[i]
            Tkey = f"{T1:.3f}"  # 保留三位小数

            # PGA/PGV特殊处理
            if Tkey == "0.000":  # PGA行
                Tkey = "-1.000"
                self.periods[i] = -1.0
            elif Tkey == "-1.000":  # PGV行
                Tkey = "-2.000"
                self.periods[i] = -2.0

            # 创建系数字典
            self.Coefs[Tkey] = {}
            for ikey, key in enumerate(self.CoefKeys):
                self.Coefs[Tkey][key] = coefs[i, ikey]  # 避免使用exec

    def __call__(
        self,
        M,
        Rjb,
        Vs30,
        T,
        rake,
        Ftype=None,
        Rrup=None,
        Rx=None,
        Ry0=None,
        dip=None,
        Ztor=None,
        Z10=None,
        W=None,
        Zhypo=None,
        azimuth=None,
        Fhw=None,
        Fas=0,
        CRjb=15,
        VsFlag=1,
        region="CA",
        country="California",
        CoefTerms={"terms": (1, 1, 1, 1, 1, 1, 1), "NewCoefs": None},
    ):

        # input Z10 should be in meter !!!
        # CRjb is defined for aftershock, if you set Fas = 0, then
        # this will not be used

        self.M = M  # moment magnitude
        self.Rjb = Rjb  # JB distance (km)
        self.Vs30 = Vs30  # site condition (m/s)
        self.rake = rake  # rake andgle
        self.country = country
        self.region = region

        if T in self.periods:
            self.T = T
        else:
            print("T is not in periods list, try to interpolate")
            raise ValueError

        self.c = 2.4 * (self.T != -2) + 2400 * (self.T == -2)

        terms = CoefTerms["terms"]
        NewCoefs = CoefTerms["NewCoefs"]

        # Obtain optional parameters
        if Ftype != None:
            self.Fnm = 1 * (Ftype == "NM")
            self.Frv = 1 * (Ftype == "RV")
        else:
            if rake == None or rake < -180 or rake > 180.0:
                print("rake angle should be within [-180,180]")
                raise ValueError
            else:
                self.Frv, self.Fnm = rake2ftype_AS(self.rake)

        if W == None:
            if self.rake == None:
                print("you should give either the fault width W " "or the rake angle")
                raise ValueError
            else:
                self.W = calc_W(self.M, self.rake)
        else:
            self.W = W

        if dip == None:
            if self.rake == None:
                print("you should give either the fault dip " "angle or the rake angle")
                raise ValueError
            else:
                self.dip = calc_dip(self.rake)
        else:
            self.dip = dip

        if Ztor == None:
            if Zhypo == None:
                if self.rake == None:
                    print("you should give either the Ztor or the rake angle")
                    raise ValueError
                else:
                    Zhypo = calc_Zhypo(self.M, self.rake)
            self.Ztor = calc_Ztor(W, self.dip, Zhypo)
        else:
            self.Ztor = Ztor

        self.azimuth = azimuth  # use the original one if available

        if Fhw == None:
            if azimuth == None and Rx == None:
                print("either one of azimuth angle, Rx and Fhw " "has to be specified")
                raise ValueError

            if azimuth != None:
                if 0 <= azimuth <= 180.0 and dip != 90.0:
                    Fhw = 1
                else:
                    Fhw = 0

            elif Rx != None:
                if Rx >= 0 and dip != 90.0:
                    Fhw = 1
                else:
                    Fhw = 0

            if dip == 90:
                Fhw = 0

        if azimuth == None:
            if Fhw == 1:
                azimuth = 50
            else:
                azimuth = -50.0

        self.Fhw = Fhw

        # Compute Rrup and Rx
        if azimuth == 90.0:
            Rx = Rrup / np.sin(self.dip * np.pi / 180.0) - Ztor / np.tan(self.dip * np.pi / 180.0)
        elif azimuth > 0.0:
            Rx = Rjb * np.tan(azimuth * np.pi / 180.0)
        elif azimuth <= 0.0:
            Rx = 0.0
        if Rx == None:
            self.Rx = calc_Rx(self.Rjb, self.Ztor, self.W, self.dip, azimuth, Rrup)
        else:
            self.Rx = Rx
        if Rrup == None:
            self.Rrup = calc_Rrup(self.Rx, self.Ztor, self.W, self.dip, azimuth, self.Rjb)
        else:
            self.Rrup = Rrup

        if Ry0 == None:
            if self.azimuth != None and self.Rx != None:
                self.Ry0 = self.Rx * np.tan(self.azimuth * np.pi / 180.0)
            else:
                self.Ry0 = None
        else:
            self.Ry0 = Ry0  # attention here (Ry0)

        # Z10
        if Z10 == None:
            if country == "Japan":
                self.Z10 = np.exp(
                    -5.23 / 2.0 * np.log((Vs30**2 + 412.0**2) / (1360.0**2 + 412.0**2))
                )
            else:
                self.Z10 = np.exp(
                    -7.67 / 4.0 * np.log((Vs30**4 + 610.0**4) / (1360.0**4 + 610.0**4))
                )
        else:
            self.Z10 = Z10

        # for ASK14, the Z10 used in calculation is in km
        self.Z10 = self.Z10 / 1000.0
        self.Fas = Fas  # aftershock flag (0 or 1)
        self.CRjb = CRjb
        self.VsFlag = VsFlag  # 0: estimated Vs30; 1: measured Vs30

        # update coeficient
        if NewCoefs != None:
            NewCoefKeys = NewCoefs.keys()
            Tkey = GetKey(self.T)
            for key in NewCoefKeys:
                self.Coefs[Tkey][key] = NewCoefs[key]

        # Compute IM and uncertainties
        IM = self.compute_im(terms=terms)
        sigma, tau, sigmaT = self.calc_sigma_tau()

        return IM, sigmaT, tau, sigma

    def base_model(self, Tother=None):
        # Basically, this is the distance-magnitude term
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)

        c4 = self.Coefs[str(Ti)]["c4"]
        a1 = self.Coefs[str(Ti)]["a1"]
        a2 = self.Coefs[str(Ti)]["a2"]
        a3 = self.Coefs[str(Ti)]["a3"]
        a6 = self.Coefs[str(Ti)]["a6"]
        a8 = self.Coefs[str(Ti)]["a8"]
        a17 = self.Coefs[str(Ti)]["a17"]
        M1 = self.Coefs[str(Ti)]["M1"]

        c4M = (
            c4 * (self.M > 5)
            + (c4 - (c4 - 1) * (5 - self.M)) * (4 < self.M <= 5)
            + 1 * (self.M <= 4)
        )
        Rtmp = np.sqrt(self.Rrup**2 + c4M**2)
        if self.M < self.M2:
            output = (
                a1
                + self.a4 * (self.M2 - M1)
                + a8 * (8.5 - self.M2) ** 2
                + a6 * (self.M - self.M2)
                + self.a7 * (self.M - self.M2) ** 2
                + (a2 + a3 * (self.M2 - M1)) * np.log(Rtmp)
                + a17 * self.Rrup
            )
        elif self.M2 <= self.M < M1:
            output = (
                a1
                + self.a4 * (self.M - M1)
                + a8 * (8.5 - self.M) ** 2
                + (a2 + a3 * (self.M - M1)) * np.log(Rtmp)
                + a17 * self.Rrup
            )
        elif self.M >= M1:
            output = (
                a1
                + self.a5 * (self.M - M1)
                + a8 * (8.5 - self.M) ** 2
                + (a2 + a3 * (self.M - M1)) * np.log(Rtmp)
                + a17 * self.Rrup
            )
        # print 'f_base=', output
        return output

    def flt_function(self, Tother=None):
        # fault type and aftershock flag
        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)

        a11 = self.Coefs[Ti]["a11"]
        a12 = self.Coefs[Ti]["a12"]
        a14 = self.Coefs[Ti]["a14"]

        f7 = a11 * (self.M > 5) + a11 * (self.M - 4) * (4 < self.M <= 5) + 0 * (self.M <= 4)
        f8 = a12 * (self.M > 5) + a12 * (self.M - 4) * (4 < self.M <= 5) + 0 * (self.M <= 4)
        f11 = (
            a14 * (self.CRjb <= 5)
            + a14 * (1 - (self.CRjb - 5) / 10.0) * (5 < self.CRjb <= 15)
            + 0 * (self.CRjb > 15)
        )

        output = self.Frv * f7 + self.Fnm * f8 + self.Fas * f11
        # print 'f_flt=', output
        return output

    def ztor_function(self, Tother=None):
        # depth to top of rupture model

        if Tother != None:
            Ti = GetKey(Tother)
        else:
            Ti = GetKey(self.T)

        a15 = self.Coefs[Ti]["a15"]

        if self.Ztor < 20:
            output = a15 * self.Ztor / 20.0
        else:
            output = a15
        # print 'f_ztor=', output
        return output

    def hw_function(self, Tother=None):
        # hanging wall function
        if self.Rx < 0:
            output = 0.0
        else:
            if Tother != None:
                Ti = GetKey(Tother)
            else:
                Ti = GetKey(self.T)
            a13 = self.Coefs[Ti]["a13"]

            # taper1
            if self.dip > 30:
                taper1 = (90 - self.dip) / 45.0
            else:
                taper1 = 60.0 / 45.0

            # taper2
            if self.M >= 6.5:
                taper2 = 1 + self.a2HW * (self.M - 6.5)
            elif 5.5 < self.M < 6.5:
                taper2 = 1 + self.a2HW * (self.M - 6.5) - (1 - self.a2HW) * (self.M - 6.5) ** 2
            else:
                taper2 = 0.0

            # taper3 (constrain the hanging wall effects, this should
            # decreasing the difference between simulation and NGA
            # GMPEs)
            R1 = self.W * np.cos(self.dip * np.pi / 180.0)
            R2 = 3 * R1
            h1 = 0.25
            h2 = 1.5
            h3 = -0.75
            if self.Rx < R1:
                taper3 = h1 + h2 * (self.Rx / R1) + h3 * (self.Rx / R1) ** 2
            elif R1 <= self.Rx <= R2:
                taper3 = 1 - (self.Rx - R1) / (R2 - R1)
            else:
                taper3 = 0.0

            # taper4 (constrain based on ztor)
            if self.Ztor <= 10:
                taper4 = 1 - self.Ztor**2 / 100.0
            else:
                taper4 = 0.0

            # taper5 (constrain azimuthally)
            Ry1 = self.Rx * np.tan(20 * np.pi / 180.0)
            if self.Ry0 != None:
                if self.Ry0 < Ry1:
                    taper5 = 1.0
                elif self.Ry0 - Ry1 < 5:
                    taper5 = 1 - (self.Ry0 - Ry1) / 5.0
                elif self.Ry0 - Ry1 > 5:
                    taper5 = 0.0
            else:
                taper5 = (
                    1.0 * (self.Rjb == 0)
                    + (1 - self.Rjb / 30.0) * (self.Rjb < 30 and self.Rjb != 0)
                    + 0.0 * (self.Rjb >= 30.0)
                )
            # print 'taper1, taper2, taper3, taper4, taper5, a13:',
            # taper1, taper2, taper3, taper4, taper5, a13
            output = a13 * taper1 * taper2 * taper3 * taper4 * taper5
        # print 'Rx, f_hng=', self.Rx, output
        return output

    def calc_MeanZ10(self, Vs30=None, country="California"):
        if Vs30 == None:
            Vs30 = self.Vs30

        if country == "Japan":
            lnZ10 = -5.23 / 2.0 * np.log((Vs30**2 + 412.0**2) / (1360.0**2 + 412.0**2))
        else:
            lnZ10 = -7.67 / 4.0 * np.log((Vs30**4 + 610.0**4) / (1360.0**4 + 610.0**4))
        MeanZ10 = np.exp(lnZ10) / 1000.0  # convert to km for program to use
        return MeanZ10

    def soil_function(self, Z10=None, Vs30=None, Tother=None):
        # soil depth function
        if Tother != None:
            Ti = GetKey(Tother)
            T = Tother
        else:
            Ti = GetKey(self.T)
            T = self.T

        if Z10 == None:
            Z10 = self.Z10

        if Vs30 == None:
            Vs30 = self.Vs30

        a43 = self.Coefs[Ti]["a43"]
        a44 = self.Coefs[Ti]["a44"]
        a45 = self.Coefs[Ti]["a45"]
        a46 = self.Coefs[Ti]["a46"]

        MeanZ10 = self.calc_MeanZ10(Vs30=Vs30, country=self.country)

        term = np.log((Z10 + 0.01) / (MeanZ10 + 0.01))
        #  print 'Z10, Z10hat, a43, term', Z10, MeanZ10, a43, term
        output = (
            a43 * (Vs30 <= 200.0)
            + a44 * (200 < Vs30 <= 300)
            + a45 * (300 < Vs30 <= 500)
            + a46 * (Vs30 > 500)
        )
        output = output * term
        # print 'f_soil=',output
        return output

    # compute Vs30* for soil and site effect functions
    def CalcVs30Star(self, Vs30, T):
        # compute V1 (used in soil-depth model)
        if T <= 0.5:
            V1 = 1500.0  # m/s
        elif 0.5 < T <= 3.0:
            V1 = np.exp(-0.35 * np.log(T / 0.5) + np.log(1500.0))
        elif T >= 3.0:
            V1 = 800.0

        # calculate Vs30*
        if Vs30 < V1:
            Vs30_1 = Vs30
        else:
            Vs30_1 = V1

        return V1, Vs30_1

    def site_model(self, SA1100, Vs30=None, Tother=None):
        # Site-response model

        if Tother != None:
            Ti = GetKey(Tother)
            T = Tother
        else:
            Ti = GetKey(self.T)
            T = self.T

        a10 = self.Coefs[Ti]["a10"]
        b = self.Coefs[Ti]["b"]
        Vlin = self.Coefs[Ti]["VLIN"]

        if Vs30 == None:
            Vs30 = self.Vs30

        V1, Vs30_1 = self.CalcVs30Star(Vs30, T)

        if Vs30 < Vlin:
            output = (
                a10 * np.log(Vs30_1 / Vlin)
                - b * np.log(SA1100 + self.c)
                + b * np.log(SA1100 + self.c * (Vs30_1 / Vlin) ** self.n)
            )
        else:
            output = (a10 + b * self.n) * np.log(Vs30_1 / Vlin)
        # print 'f_site=',output
        return output

    def SA1100_calc(self):
        # compute SA1100 (different from AS08)
        SA1100Rock = 0.0
        Vs30Rock = 1100.0
        Z10Rock = calc_Z1(Vs30Rock, "AS") / 1000.0  # attention here
        Tother = self.T  # SA at current period !  cout << "Z10Rock: "
        # << Z10Rock << ", Z10hat: " << Z10hat << ", a46: " <<
        # s_a46[iT] << ", term: " << tmp << endl;
        SA1100 = (
            self.base_model(Tother=Tother)
            + self.flt_function(Tother=Tother)
            + self.site_model(SA1100Rock, Vs30=Vs30Rock, Tother=Tother)
            + self.Fhw * self.hw_function(Tother=Tother)
            + self.ztor_function(Tother=Tother)
            + self.soil_function(Z10=Z10Rock, Vs30=Vs30Rock, Tother=Tother)
        )
        output = np.exp(SA1100)
        return output

    # def RegionalCorrection(self, Vs30=None, Rrup=None, Tother=None):
    #     if Tother != None:
    #         Ti = GetKey(Tother)
    #         T = Tother
    #     else:
    #         Ti = GetKey(self.T)
    #         T = self.T
    #     if Vs30 == None:
    #         Vs30 = self.Vs30
    #     if Rrup == None:
    #         Rrup = self.Rrup

    #     for key in ["VLIN", "a31", "a28", "a29", "a36", "a37", "a38", "a40", "a41", "a42"]:
    #         cmd = "%s = self.Coefs['%s']['%s']" % (key, Ti, key)
    #         exec(cmd)

    #     if self.region == "CA":
    #         return 0.0
    #     elif self.region == "TW":
    #         f11 = a31 * np.log(Vs30 / Vlin)
    #         return f11 + a25 * Rrup
    #     elif self.region == "CN":
    #         return a28 * Rrup
    #     elif self.region == "JP":
    #         f12 = (
    #             a36 * (Vs30 < 200)
    #             + a37 * (200 <= Vs30 < 300)
    #             + a38 * (300 <= Vs30 < 400)
    #             + self.a39 * (400 <= Vs30 < 500)
    #             + a40 * (500 <= Vs30 < 700)
    #             + a41 * (700 <= Vs30 < 1000)
    #             + a42 * (Vs30 >= 1000)
    #         )
    #         return f12 + a29 * Rrup

    def RegionalCorrection(self, Vs30=None, Rrup=None, Tother=None):
        # 参数初始化
        if Tother is not None:  # 更规范的None判断
            Ti = GetKey(Tother)
            T = Tother
        else:
            Ti = GetKey(self.T)
            T = self.T

        Vs30 = self.Vs30 if Vs30 is None else Vs30
        Rrup = self.Rrup if Rrup is None else Rrup

        # 安全获取系数
        Coefs_Ti = self.Coefs[Ti]
        required_coefs = [
            "VLIN",
            "a25",
            "a28",
            "a29",
            "a31",
            "a36",
            "a37",
            "a38",
            "a40",
            "a41",
            "a42",
        ]
        # ["VLIN", "a31", "a28", "a29", "a36", "a37", "a38", "a40", "a41", "a42"]

        # 确保所有需要的系数都存在
        for coef in required_coefs:
            if coef not in Coefs_Ti:
                raise KeyError(f"Missing coefficient {coef} for period {Ti}")

        # 显式赋值（更安全可读）
        VLIN = Coefs_Ti["VLIN"]
        a25 = Coefs_Ti["a25"]
        a28 = Coefs_Ti["a28"]
        a29 = Coefs_Ti["a29"]
        a31 = Coefs_Ti["a31"]
        a36 = Coefs_Ti["a36"]
        a37 = Coefs_Ti["a37"]
        a38 = Coefs_Ti["a38"]
        a40 = Coefs_Ti["a40"]
        a41 = Coefs_Ti["a41"]
        a42 = Coefs_Ti["a42"]

        # 地区逻辑
        if self.region == "CA":
            return 0.0
        elif self.region == "TW":
            f11 = a31 * np.log(Vs30 / VLIN)  # 修复大小写
            return f11 + a25 * Rrup
        elif self.region == "CN":
            return a28 * Rrup
        elif self.region == "JP":
            # 更清晰的逻辑判断
            vs30_conditions = [
                (Vs30 < 200, a36),
                (200 <= Vs30 < 300, a37),
                (300 <= Vs30 < 400, a38),
                (400 <= Vs30 < 500, self.a39),
                (500 <= Vs30 < 700, a40),
                (700 <= Vs30 < 1000, a41),
                (Vs30 >= 1000, a42),
            ]
            f12 = sum(value * cond for cond, value in vs30_conditions if cond)
            return f12 + a29 * Rrup
        else:
            raise ValueError(f"Unsupported region: {self.region}")

    # function to compute the intensity
    def logline(self, x1, x2, y1, y2, x):
        # linear interpolation
        k = (y2 - y1) / (x2 - x1)
        C = y1 - k * x1
        y = k * x + C
        return y

    def compute_im(self, terms=(1, 1, 1, 1, 1, 1, 1)):

        # print 'Compute SA1100_calc'
        SA1100 = self.SA1100_calc()
        # print 'SA1100=',SA1100
        # print '===================================='
        LnSa = (
            terms[0] * self.base_model()
            + terms[1] * self.flt_function()
            + terms[2] * (self.Fhw * self.hw_function() + terms[3] * self.ztor_function())
            + terms[4] * self.site_model(SA1100)
            + terms[5] * self.soil_function()
            + terms[6] * self.RegionalCorrection()
        )
        IM = np.exp(LnSa)
        # print 'IM = ',IM
        return IM

    # compute standard deviations
    def calc_alpha(self):
        Ti = GetKey(self.T)
        Vlin = self.Coefs[Ti]["VLIN"]
        b = self.Coefs[Ti]["b"]

        SA1100 = self.SA1100_calc()
        if self.Vs30 >= Vlin:
            alpha = 0.0
        else:
            alpha = -b * SA1100 / (SA1100 + self.c) + b * SA1100 / (
                SA1100 + self.c * (self.Vs30 / Vlin) ** self.n
            )
        # print 'Alpha=',alpha
        return alpha

    def calc_sigma_tau(self):
        Ti = GetKey(self.T)
        if self.country != "Japan":
            if self.VsFlag == 0:
                s1 = self.Coefs[Ti]["s01"]
                s2 = self.Coefs[Ti]["s02"]
            if self.VsFlag == 1:
                s1 = self.Coefs[Ti]["s11"]
                s2 = self.Coefs[Ti]["s12"]
            phi_AL = (
                s1 * (self.M < 4)
                + (s1 + (s2 - s1) / 2 * (self.M - 4)) * (4 <= self.M <= 6)
                + s2 * (self.M > 6)
            )
        else:
            s5 = self.Coefs[Ti]["s5"]
            s6 = self.Coefs[Ti]["s6"]
            phi_AL = (
                s5 * (self.Rrup < 30)
                + (s5 + (s6 - s5) / 50.0 * (self.Rrup - 30)) * (30 <= self.Rrup <= 80)
                + s6 * (self.Rrup > 80)
            )

        s3 = self.Coefs[Ti]["s3"]
        s4 = self.Coefs[Ti]["s4"]
        tau_AL = (
            s3 * (self.M < 5)
            + (s3 + (s4 - s3) / 2.0 * (self.M - 5)) * (5 <= self.M < 7)
            + s4 * (self.M >= 7)
        )

        phi_Amp = 0.4
        alpha = self.calc_alpha()
        phi_B = np.sqrt(phi_AL**2 - phi_Amp**2)
        sigma = np.sqrt((phi_B * (1 + alpha)) ** 2 + phi_Amp**2)

        tau_B = tau_AL
        tau = tau_B * (1 + alpha)

        sigmaT = np.sqrt(sigma**2 + tau**2)
        return (sigma, tau, sigmaT)


def ASK14nga_test(T, CoefTerms):
    """
    Test AS nga model
    """
    Mw = 8.0
    Zhypo = 8.0
    Ztor = 0.0
    dip = 90
    Ftype = "SS"
    rake = 0  # for specific rupture
    W = 100

    Rjb = 3.0
    Rrup = Rjb
    Rx = Rrup
    # Rrup = (W*np.sin(dip*np.pi/180.)+Ztor) * np.cos(dip*np.pi/180.)
    # Rx = W*np.cos(dip*np.pi/180.)

    # print "Rx", Rx
    # Rx = Rrup

    Vs30 = 748.0, 1200.0, 345.0, 160.0
    Vs30 = 760.0
    Z25 = Z10 = None

    Fas = 0
    VsFlag = 0

    ASKnga = ASK14_nga()

    kwds = {
        "Ftype": Ftype,
        "Rrup": Rrup,
        "Rx": Rx,
        "dip": dip,
        "Ztor": Ztor,
        "W": W,
        "Z10": Z10,
        "Fas": Fas,
        "VsFlag": VsFlag,
        "CoefTerms": CoefTerms,
    }
    values = mapfunc(ASKnga, Mw, Rjb, Vs30, T, rake, **kwds)
    print("Median, SigmaT, Tau, Sigma")
    for i in range(len(values)):
        print(values[i])

    return ASKnga


if __name__ == "__main__":
    if 0:
        # SA test
        # T = 0.1; NewCoefs = {'Vlin':500,'b':-1.024}
        # T = 0.1; NewCoefs = None
        # T = 0.1; NewCoefs = {'Vlin':1032.5,'b':-1.624}
        NewCoefs = None
        T = 1.0
        CoefTerms = {"terms": (1, 1, 1, 1, 1, 1, 1), "NewCoefs": NewCoefs}

        print("AS SA at %s" % ("%3.2f" % T))
        AS14nga = AS14nga_test(T, CoefTerms)

        T = -1.0
        print("AS PGA:")
        CoefTerms = {"terms": (1, 1, 1, 1, 1, 1, 1), "NewCoefs": None}
        AS14nga = ASK14nga_test(T, CoefTerms)
    else:
        # Notes: PGV for Vs30 = 760, Z10 = 24, the soil-depth function
        # should be the same as T=1.0
        # for T in [-1.0,-2.0, 0.01, 0.02,
        # 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
        # 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]:
        for T in [
            -1.0,
        ]:
            print("AS SA at %s" % ("%3.2f" % T))
            CoefTerms = {"terms": (1, 1, 1, 1, 1, 1, 1), "NewCoefs": None}
            ASKnga = ASK14nga_test(T, CoefTerms)
