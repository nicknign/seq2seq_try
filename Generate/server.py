# -*- coding: utf-8 -*-
# !/usr/bin/env python
import pika
import sys
import json

import manage
from macro import *

reload(sys)
sys.setdefaultencoding('utf8')


class Server(object):
    def __init__(self, rabbitmq):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=rabbitmq))
        self.channel = self.connection.channel()
        self.channel.queue_declare()
        self.manager = manage.Manager()
        self.name = "Retrival"

    def handle(self, msg):
        inputdict = json.loads(msg)
        userid = inputdict["userid"]
        groupid = inputdict["groupid"]
        mode = inputdict["mode"]
        bot = self.manager.get_create_bot(userid, groupid, mode)
        output, score = bot.sentences_respond(inputdict["input"])
        outputdict = {"name": self.name, "userid": userid, "output": output, "score": score}
        answerjson = json.dumps(outputdict)
        return answerjson

    def on_request(self, ch, method, properties, body):
        response = self.handle(body)
        try:
            ch.basic_publish(exchange='',
                             routing_key=properties.reply_to,  # answer queue
                             properties=pika.BasicProperties(correlation_id=
                                                             properties.correlation_id),
                             body=unicode(response))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.critical("[Retrival]exception: " + unicode(Exception) + ":" + unicode(e))
            log.log_traceback()

        logger.debug("[Retrival]send rpc back, routing_key:" + properties.reply_to)
        logger.debug("[Retrival]send rpc back, cor_id:" + properties.correlation_id)

    def start(self):
        result = self.channel.queue_declare(exclusive=True)
        queue_name = result.method.queue
        self.channel.queue_bind(exchange='input',
                                queue=queue_name)
        self.channel.basic_consume(self.on_request,
                                   queue=queue_name)
        logger.debug("[Retrival]Awaiting RPC requests")
        self.channel.start_consuming()


if __name__ == "__main__":
    pass
