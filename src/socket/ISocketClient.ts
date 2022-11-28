import { Socket } from 'socket.io-client';
interface IServerToClientEvents {
  message:  any;
}
interface IClientToServerEvents {
  connect: any;
  message:  any;
}


type SocketType = Socket<IServerToClientEvents, IClientToServerEvents>;

export type {SocketType};