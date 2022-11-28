import { SocketType } from './ISocketClient';

import create from 'zustand';
import React, { Fragment } from 'react';
import { io } from 'socket.io-client';
interface ISocketState {
  socket: SocketType | null;
  setSocket: (socket: SocketType) => void;
}

const BackendUrl = 'http://127.0.0.1:5000';

const useAppSocket = create<ISocketState>()(setState => ({
  socket: null,
  setSocket: socket => setState({ socket }),
}));
const SocketWrapper = ({ children }: { children: React.ReactNode }) => {
  const { setSocket } = useAppSocket();

  React.useEffect(() => {
    try {
      // const options: Partial<ManagerOptions & SocketOptions> = {
      //   transports: ['websocket', 'polling'],
      //   closeOnBeforeunload: false,
      // };
      const socketCnn: SocketType = io(BackendUrl);
      setSocket(socketCnn);
    } catch (e) {
      console.log(e);
    }
  }, [setSocket]);

  return <Fragment>{children}</Fragment>;
};

const useSocket = () => {
  const socketSelector = React.useCallback((state: ISocketState) => {
    return state.socket;
  }, []);
  return useAppSocket(socketSelector);
};

export default SocketWrapper;
export { useSocket };
