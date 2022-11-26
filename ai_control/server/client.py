import socket

SERVER_ADDRESS = '169.254.161.148'
PORT = 51001

def createClient():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    try:
        soc.connect((SERVER_ADDRESS, PORT))
        try:
            model = 'mobi-v2'
            soc.sendall(model.encode())
            while True:
                file_name = input("Enter: ")
                f = open("../img_test/" + file_name + '.txt', "r")
                soc.sendall(f.read().encode())
                result = soc.recv(1024).decode()
                print(result)
        finally:
            soc.close()
    finally:
        soc.close()

createClient()