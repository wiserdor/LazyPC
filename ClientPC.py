from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import time
import sys
import requests


class ClientPC:
    def __init__(self):
        print("starting RPIClient...")
        self.connected = False

    def receive(self):
        """ Handles receiving messages. """
        while True:
            try:
                msg = self.sock.recv(1024).decode("utf-8")
                # print(msg,'\n')
                ##                if "Connection is now available" in msg:
                ##                    self.connected=True
                time.sleep(8)
                self.connected = True
            except OSError:  # Possibly client has left the chat.
                break

    def isConnected(self):
        return self.connected

    def send(self, msg):
        """ Handles sending messages. """
        if self.connected:
            self.sock.sendall(msg.encode() + '\r\n'.encode())
            if msg == "!!@@quit":
                self.sock.close()
                self.connect()
            print('sent:', msg, '\n')
        else:
            print("not connected")

    def connect(self):
        print("Trying to connect...")
        ip = '193.106.55.107'
        port = 5000
        ADDR = (ip, port)
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect(ADDR)
        receive_thread = Thread(target=self.receive)
        receive_thread.start()
        print('Connected to server: ', ip, ':', port, 'successfully')
        self.sock.sendall('segev@gmail.com 123 rasp'.encode() + '\r\n'.encode())

#
# s=RPIClient()
# s.connect()