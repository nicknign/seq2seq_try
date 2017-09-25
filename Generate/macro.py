# -*- coding: utf-8 -*
import ConfigParser
import os
filepath = os.path.split(os.path.realpath(__file__))[0]

RABBIT_URL = os.environ.get('RABBITMQ_URL')
if not RABBIT_URL:
    RABBIT_URL = "amqp://robot:robot@amqp:5672/robot"

# DB Attr Key_value
MODEL = "modeldesc"
PRICE = "factoryprice"
FACTORY = "factory"
GRADE = "grade"
SEATNUM = "seatnum"
SHOWPARA = "showpara"
NUMBER = "number"

# dialog key
SEARCHFIN = "SEARCHFIN"
QUERYFIN = "QUERYFIN"
MULTIKEY = "MULTIKEY"
CARNUM = "CARNUM"
MULTIRESULT = "MULTIRESULT"
RESTART = "RESTART"
RESULTOOMUCH = "RESULTOOMUCH"
NOSUPORTPARA = "NOSUPORTPARA"
ISANS = "ISANS"
ISYAN = "ISYAN"
ISMULTI = "ISMULTI"
MULTIPRICE = "MULTIPRICE"
COMMENT = "COMMENT"
COMMENTASK = "COMMENTASK"
NOCOMMENT = "NOCOMMENT"
COMPAREDIFF = "COMPAREDIFF"
COMPARESAME = "COMPARESAME"

CMD = ["WELCOME", "QUERY", "SEARCH", "FUZZYQUERY", "SEARCHQUERY",
       "COMPARE", "PARA", "TERMINOLOGY", "ISQUERY"]

DIALOG = {
    FACTORY: "ASK FACTORY NUM * VALUE *",
    MODEL: "ASK MODEL NUM * VALUE *",
    PRICE: "ASK PRICE",
    GRADE: "ASK GRADE NUM * VALUE *",
    SEATNUM: "ASK SEATNUM NUM * VALUE *",
    SEARCHFIN: "SEARCH FIN NUM * DESC *",
    QUERYFIN: "QUERY FIN KEY * VALUE *",
    MULTIKEY: "MULTIKEY *",
    CARNUM: "CARNUM *",
    MULTIRESULT: "MULTIRESULT MODEL * KEY * VALUE *",
    MULTIPRICE: "MULTIPRICE MODEL * VALUE *",
    RESTART: "RESTART COND *",
    RESULTOOMUCH: "RESULTOOMUCH",
    NOSUPORTPARA: "NOSUPPORT PARA *",
    ISANS: "ISANS ANS * PARA *",
    ISYAN: "ISYAN YES * NO *",
    ISMULTI: "ISMULTI MODEL * ANS * PARA *",
    COMMENT: "COMMENT CAR * COMMENT *",
    COMMENTASK: "COMMENT ASK",
    NOCOMMENT: "NOCOMMENT",
    COMPAREDIFF: "COMPARE SAME * DIFF *",
    COMPARESAME: "COMPARE SAME",
}

TRANSLATE = {
    FACTORY: u"厂商",
    MODEL: u"车型",
    PRICE: u"价格",
    GRADE: u"级别",
    SEATNUM: u"座位数",
    u"rfuelconsumption": u"油耗",
    u"engine": u"发动机",
    u"transmisson": u"变速箱",
    u"length": u"长度",
    u"displacementL": u"排量",
    u"maxspeed": u"最高车速",
    u"alr100kmh": u"官方百公里加速",
    u"maxhorsepower": u"最大马力",
    u"colour": u"颜色",
    u"automaticparkinplace": u"自动停车功能",
    u"ISOFIXchildseatinterface": u"ISOFIX儿童安全座椅接口",
    u"rearindependentaircondition": u"后排独立空调",
    u"lenwidhig": u"长宽高",
    u"bodystruc1": u"车身结构",
    u"ralr100kmh": u"实测百公里加速",
    u"rbrake100kmh": u"实测百公里制动",
    u"gxbtotalconsumption": u"工信部综合油耗",
    u"rlidijianxi": u"离地间隙",
    u"autowarranty": u"整车质保",
    u"width": u"宽度",
    u"height": u"宽度",
    u"wheelbase": u"轴距",
    u"fronttrack": u"前轮距",
    u"reartrack": u"后轮距",
    u"minlidijianxi": u"最小离地间隙",
    u"curbquality": u"整备质量",
    u"doornum": u"车门数",
    u"fueltankcapacity": u"油箱容积",
    u"luggagecapacity": u"行李箱容量",
    u"enginemodel": u"发动机型号",
    u"nightvisionsys": u"夜视系统",
    u"leathersteerwheel": u"真皮方向盘",
    u"seatmaterial": u"座椅材质",
}

# TODO: create a log class for more detailed log controller
LOG_SHOW = False

ARGORDER = [{FACTORY: "", MODEL: "", GRADE: "", PRICE: ""},  # first concern
            {SEATNUM: ""}]

PLAIN = [FACTORY, MODEL, GRADE]

EXTRACT_MODEL_MAXNUM = 50
ANALYZE_THRESHOLD = 50
CLASSISENSI = 0.7

FILTER_STR = [u"(进口)", u"*", u"(", u")"]

SEPARATOR = u'[,，.。;；?？]'

# TODO: use tf-ide to fliter common word in the future
COMMON_WORD = [u"最远", u"最近", u"最短", u"最长", u"最高", u"最小", u"最低", u"最大", u"数",
               u"前", u"后", u"主", u"副", u"<d>", u"<f>", u"<y>", u"/", u"-", u"车", u"的",
               u"●", u"○", u"手动", u"自动", u"支持", u"显示"]

LABEL_WORD = ["terminology", "number"]

UNIT = ["L", "mL", "s", "km/h", "Ps", "L/100km"]

YES = ["●"]
NO = ["NaN", None, "nan", "○"]
ISTYPE = {
    "有": ["有", "没有"],
    "是": ["是", "不是"],
    "支持": ["支持", "不支持"],
}

# Group define
GROUP_COMMON = 0
GROUP_AUTOFORCE = 1
GROUP_4S = 2

GROUP_PATTERN = {
    GROUP_COMMON: ["pattern.csv", "common_autoforce.csv"],
    GROUP_AUTOFORCE: ["pattern_autoforce.csv", "common_autoforce.csv"],
    GROUP_4S: ["pattern_4s.csv"],
}

GROUP_DB = {
    GROUP_COMMON: ["cardb.csv", "attrdf.csv"],
    GROUP_AUTOFORCE: ["cardb.csv", "attrdf.csv"],
    GROUP_4S: ["cardb_4s.csv", "attrdf_4s.csv"],
}

GROUP_MODEL = {
    GROUP_COMMON: "common/",
    GROUP_AUTOFORCE: "autoforce/",
    GROUP_4S: "4s/",
}

COMPARE_LIST = ["grade", "seatnum", "engine", "transmisson", "maxspeed", "alr100kmh", "rfuelconsumption", "factoryprice"]

RICH = 1
TEXT = 0

# ERRCODE
OK = 0
ERR = 20000
DB_CONNECT_FAIL = 20060

