# -*- coding: utf-8 -*-
# !/usr/bin/env python

import threading

from macro import *

TIMER_LEN = 60*10  # 10 min


class Manager(object):
    def __init__(self):
        self.salebotlist = []

    def start_timer(self, userid):
        timer = threading.Timer(TIMER_LEN, self.del_bot, [userid])
        timer.setDaemon(True)
        timer.start()
        return timer

    def refresh_timer(self, parali):
        parali[1].cancel()
        userid = parali[0].userid
        timer = self.start_timer(userid)
        return timer

    def create_bot(self, userid, groupid=GROUP_COMMON, mode=TEXT):
        bot = salebot.Salebot(userid, groupid, mode)
        timer = self.start_timer(userid)
        self.salebotlist.append([bot, timer])
        logger.info("[Retrival][Manager]create bot of userid:{}, groupid:{}, mode:{}".format(userid, groupid, mode))
        return bot

    def get_create_bot(self, userid, groupid=GROUP_COMMON, mode=TEXT):
        for bot in self.salebotlist:
            if bot[0].userid == userid:
                bot[1] = self.refresh_timer(bot)
                logger.info("[Retrival][Manager]get bot userid:" + userid)
                return bot[0]
        bot = self.create_bot(userid, groupid, mode)
        return bot

    def del_bot(self, userid):
        for bot in self.salebotlist:
            if bot[0].userid == userid:
                self.salebotlist.remove(bot)
                del bot
                logger.info("[Retrival][Manager]delete bot of userid:{}".format(userid))
                return True
        logger.warning("[Retrival][Manager]there is no bot of userid {}".format(userid))
        return False
